import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class Loss():
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer

    def stable_softmax(self, logits):
            logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
            logits = logits.clip(min=-10)
            return torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True)

    def compute_value(self, Q_value, projected_policy, mask, response_ids):
        value = torch.sum(
            projected_policy * (Q_value - torch.log(projected_policy)),
            dim=2
        ) * mask
        return value
 
    def compute_projected_policy(self, Q_value, teacher_top_p_mask):
        student_policy = self.stable_softmax(Q_value)
        projected_policy = student_policy * teacher_top_p_mask / self.args.top_p + self.args.epsilon * torch.ones(student_policy.shape).to(student_policy.device)
        projected_policy = projected_policy / torch.sum(projected_policy, dim=2, keepdim=True)

        return projected_policy

    def bellman_distill_loss(self, batch, student_logits, teacher_logits, teacher_top_p_mask, student_state_value=None, gamma=0.99):

        stats = {}
        query_tensors = batch["query_tensors"]
        teacher_responses = batch["teacher_responses"]
        mask = (batch["teacher_responses"] != self.trainer.tokenizer.pad_token_id)

        inf_mask_student = torch.isinf(student_logits)
        response_length = teacher_responses.shape[-1]

        start = query_tensors.size(1) - 1
        end = query_tensors.size(1) + teacher_responses.size(1) - 1
        student_logits = student_logits / self.args.temperature

        if inf_mask_student is not None:
            student_logits = student_logits.masked_fill(inf_mask_student, -float("inf"))

        stats["mask"] = mask
        # pdb.set_trace()
        Q_value = student_logits
        Q_value = student_logits - torch.max(student_logits, dim=-1, keepdim=True)[0]
        Q_value = Q_value.clip(min=-10)
        projected_policy = self.compute_projected_policy(Q_value, teacher_top_p_mask)
        V_value = self.compute_value(Q_value, projected_policy, mask, teacher_responses)

        with torch.no_grad():
            Q_value_test = student_logits - torch.max(student_logits, dim=-1, keepdim=True)[0]
            Q_value_test = student_logits
            Q_value_test = Q_value_test.clip(min=-10)
            projected_policy_test = self.compute_projected_policy(Q_value_test, teacher_top_p_mask)
            V_value_test = self.compute_value(Q_value_test, projected_policy_test, mask, teacher_responses)
            V_next_value_mean_test = V_value_test[:, 1:] * mask[:, 1:]
            track_value = torch.sum(torch.mean(teacher_top_p_mask[:, :-1, :] * Q_value_test[:, :-1, :] * mask[:, :-1].unsqueeze(-1) \
                            - gamma * V_next_value_mean_test.unsqueeze(-1), dim=2)) / mask[:, :-1].sum()

        V_next_value_mean = V_value[:, 1:] * mask[:, 1:]
        if self.args.chi_reg:
            loss = -(torch.sum(torch.mean(teacher_top_p_mask[:, :-1, :] * Q_value[:, :-1, :] * mask[:, :-1].unsqueeze(-1) \
                        - gamma * V_next_value_mean.unsqueeze(-1), dim=2)) / mask[:, :-1].sum() \
                        - torch.sum(V_value[:, :-1] * mask[:, :-1] \
                        - gamma * V_next_value_mean) / mask[:, :-1].sum())
            loss += (1 / (4 * 0.1)) * torch.sum(torch.mean((teacher_top_p_mask[:, :-1, :] * Q_value[:, :-1, :] * mask[:, :-1].unsqueeze(-1) \
                        - gamma * V_next_value_mean.unsqueeze(-1)) ** 2, dim=2)) / mask[:, :-1].sum()
        else:
            loss = -(torch.sum(torch.mean(teacher_top_p_mask[:, :-1, :] * Q_value[:, :-1, :] * mask[:, :-1].unsqueeze(-1) \
                        - gamma * V_next_value_mean.unsqueeze(-1), dim=2)) / mask[:, :-1].sum() \
                        - torch.sum(V_value[:, :-1] * mask[:, :-1] \
                        - gamma * V_next_value_mean) / mask[:, :-1].sum())


        with torch.no_grad():
        
            stats["bellman_loss"] = loss.item()
            stats["Q_value"] = Q_value.mean()
            flattened_pass_Q = (teacher_top_p_mask * Q_value).flatten()
            flattened_pass_Q = flattened_pass_Q[flattened_pass_Q != 0]
            rev_mask = ~teacher_top_p_mask
            stats["Q_pass_topp"] = flattened_pass_Q.mean()
            flattened_mask_Q = (rev_mask * Q_value).flatten()
            flattened_mask_Q = flattened_mask_Q[flattened_mask_Q != 0]
            stats["Q_masked_topp"] = flattened_mask_Q.mean()

            stats["V_value"] = V_value.mean()
            stats["Q_difference_original"] = torch.sum(torch.mean(teacher_top_p_mask[:, :-1, :] * Q_value[:, :-1, :] * mask[:, :-1].unsqueeze(-1) \
                - gamma * V_next_value_mean.unsqueeze(-1), dim=2)) / mask[:, :-1].sum()
            stats["Q_difference"] = track_value
        
        return loss, stats


    def pt_loss(self, batch, logits):
        stats = {}
        model_batch, no_model_batch = batch
        loss_mask = (no_model_batch["label"] != -100).int()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
        
        distil_loss = 0
        if self.trainer.teacher_model is not None and self.args.kd_ratio is not None:
            with torch.no_grad():
                teacher_outputs = self.trainer.teacher_model(**model_batch, return_dict=True, use_cache=False)
                teacher_logits = teacher_outputs.logits
            teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
            inf_mask = torch.isinf(logits)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
            x = torch.sum(prod_probs, dim=-1).view(-1)
            distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
            
            loss = (1-self.args.kd_ratio) * lm_loss + self.args.kd_ratio * distil_loss

        stats["pt_loss"] = loss.item()
        stats["lm_loss"] = lm_loss.item()
        stats["ds_loss"] = distil_loss.item()

        return loss, stats

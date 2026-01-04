base_path="base_path"
port=$(shuf -i 20000-65000 -n 1)
eval_batch_size=32
ckpt="path/to/bellman/ckpt"


for data in dolly self_inst vicuna
do  
    # Evaluate Bellman Distill
     for seed in 10 20 30 40 50
     do
         bash ${base_path}/scripts/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size $eval_batch_size
    done
done

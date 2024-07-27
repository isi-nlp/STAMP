dataset="data/cds/sampled"
save_path="trained_models/cds/classifier"
hf_model_name="roberta-large"

learning_rate="5e-5"
n_epochs=6
batch_size=16
gradient_accumulation_steps=2

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=src

cmd="torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29475 \
    src/train_classifier.py \
    --dataset $dataset \
    --save_path $save_path \
    --hf_model_name $hf_model_name \
    --n_epochs $n_epochs \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --random_seed 42 \
    --disable_tqdm"

echo $cmd 
eval $cmd
dataset="data/paranmt"
save_path="trained_models/paraphraser"
llama_model="meta-llama/Llama-2-7b-hf"
task="paraphrase"

learning_rate="5e-5"
n_epochs=10
batch_size=32
gradient_accumulation_steps=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL

cmd="torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:50002 \
    src/sft.py \
    --dataset $dataset \
    --save_path $save_path \
    --base_model_name $llama_model \
    --task $task \
    --n_epochs $n_epochs \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --random_seed 42 \
    --disable_tqdm"

echo $cmd 
eval $cmd

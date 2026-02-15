set -x
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

usage() {
    cat <<'USAGE'
Usage: run_qwen3_vl-grpo_vllm_lora.sh [ENGINE] [DATA_DIR] [TP] [-- extra args...]

Positional args:
  ENGINE   Rollout engine name (default: vllm)
  DATA_DIR Dataset root directory (default: /scratch/arthurz/public_datasets)
  HF_MODEL_PATH HF model name or path (default: Qwen/Qwen3-VL-2B-Instruct)
  TP       Tensor parallel size (default: 1)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ENGINE=${1:-vllm}
DATA_DIR=${2:-/scratch/arthurz/public_datasets}
HF_MODEL_PATH=${3:-Qwen/Qwen3-VL-2B-Instruct}
TP=${4:-1}

# Validate TP and check GPU availability.
if ! [[ "$TP" =~ ^[0-9]+$ ]] || [ "$TP" -lt 1 ]; then
    echo "Error: TP must be a positive integer, got '$TP'." >&2
    exit 1
fi

# shift arguments so that $@ now contains any additional arguments after the first two
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift

export VLLM_WORKER_MULTIPROC_METHOD="spawn" # for vllm0.14.0 for async
export VLLM_ALLREDUCE_USE_SYMM_MEM=0        # for vllm0.14.0 
export TORCH_CUDA_ARCH_LIST="12.0"

# HPC Cluster only issue, occurs when nvc is loaded and overrides system gcc 
unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Not stable yet with lora
# actor_rollout_ref.model.lora_rank=64 \
# actor_rollout_ref.model.lora_alpha=32 \
# actor_rollout_ref.model.target_modules=all-linear \
# actor_rollout_ref.model.exclude_modules='.*visual.*' \

# actor_rollout_ref.rollout.max_num_seqs=64 \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/geo3k/train.parquet \
    data.val_files=$DATA_DIR/geo3k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.load_format='safetensors' \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

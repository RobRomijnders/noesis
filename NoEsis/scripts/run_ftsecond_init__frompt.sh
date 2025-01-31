
# CUDA_VISIBLE_DEVICES=1  python3 -m debugpy --listen 5678 main_tune.py \
datestring=`date +"%Y%m%d-%H%M%S"`

export NNUM="${NNODENUM:-0}"

SEED=130
LANGUAGE='java,python,go'
NUM_EXPERTS=3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 main_tune.py \
    --batch-size-per-replica=16 \
    --cache-data=cache_data/ \
    --ckpt-dir='saved_models/lora_dp/go-java-python_num-1_seed231_0afaun5o/final_checkpoint' \
    --datasets=${LANGUAGE} \
    --disable-dp \
    --epochs=12 \
    --finetune-second=1 \
    --freeze-backbone \
    --freeze-common \
    --load=Salesforce/codet5p-220m \
    --num-experts=${NUM_EXPERTS} \
    --num-prompt-tokens=-1 \
    --rank-common=-1 \
    --rank-domain=512 \
    --save-dir=saved_models/ftsecond_frompt_${LANGUAGE}_${NUM_EXPERTS}_${NNUM}_seed${SEED} \
    --seed=${SEED} 2>&1 | tee log/log_${datestring}_${LANGUAGE}_${NUM_EXPERTS}_torchrun.txt

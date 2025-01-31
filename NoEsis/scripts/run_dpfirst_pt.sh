
# CUDA_VISIBLE_DEVICES=1  python3 -m debugpy --listen 5678 main_tune.py \
datestring=`date +"%Y%m%d-%H%M%S"`

export NNUM="${NNODENUM:-0}"

SEED=130
LANGUAGE='java,python,go'
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 main_tune.py \
    --batch-size-per-replica=24 \
    --cache-data=cache_data/ \
    --datasets=${LANGUAGE} \
    --epochs=12 \
    --freeze-backbone \
    --load=Salesforce/codet5p-220m \
    --num-experts=-1 \
    --num-prompt-tokens=32 \
    --rank-common=-1 \
    --rank-domain=-1 \
    --save-dir=saved_models/dpfirst_ptshared_${LANGUAGE}_${NNUM}_seed${SEED} \
    --seed=${SEED} \
    --target-epsilon=1. 2>&1 | tee log/log_${datestring}_${LANGUAGE}_torchrun.txt

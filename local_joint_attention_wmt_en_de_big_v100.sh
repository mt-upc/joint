#!/bin/bash

#SBATCH --job-name=wmt14_en_de
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task 1   # Number of CPUs per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=30G           # CPU memory per node


stage=0
exp=`basename $0 | sed -e 's/^run_//' -e 's/.sh$//'`
exp=local_joint_attention_wmt_en_de_big
echo $exp

DATA=data-bin/wmt16_en_de_bpe32k
SAVE="checkpoints/$exp"
mkdir -p $SAVE

python -m torch.distributed.launch --nproc_per_node 8 $(which fairseq-train) \
    $DATA --fp16 --log-interval 100 --no-progress-bar \
    --max-update 30000 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 32 --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 1800 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.0009 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 20000 \
    --arch local_joint_attention_wmt_en_de_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.3 \
    --user-dir models

# Checkpoint averaging
python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --path "${SAVE}/checkpoint_last10_avg.pt" --batch-size 32 --beam 5 \
    --user-dir models --remove-bpe --lenpen 0.35 --gen-subset test > wmt16_gen.txt
bash scripts/compound_split_bleu.sh wmt16_gen.txt

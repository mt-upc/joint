#!/bin/bash

#SBATCH --job-name=iwslt14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1   # Number of CPUs per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G           # CPU memory per node


exp=local_joint_attention_iwslt_de_en
echo $exp

DATA=data-bin/iwslt14.joined-dictionary.31K.de-en
SAVE="checkpoints/$exp"
mkdir -p $SAVE

fairseq-train $DATA \
    --user-dir models \
    --arch local_joint_attention_iwslt_de_en \
    --clip-norm 0 --optimizer adam --lr 0.0007 \
    --source-lang de --target-lang en --max-tokens 4000 --no-progress-bar \
    --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --ddp-backend=no_c10d \
    --max-update 85000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --adam-eps '1e-09' --keep-last-epochs 10 \
    --arch local_joint_attention_iwslt_de_en --share-all-embeddings \
    --save-dir $SAVE

python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# Evaluation
fairseq-generate $DATA --path "${SAVE}/checkpoint_last10_avg.pt" --user-dir models \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 1.7 --gen-subset test --quiet

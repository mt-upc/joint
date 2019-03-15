# Joint Source-Target Self Attention with Locality Constraints (Fonollosa et al., 2019)
This repository contains the source code, pre-trained models, as well as instructions to reproduce the results or [our paper](http://www.acl2019.org/EN/index.xhtml)

## Citation:
```bibtex
@inproceedings{fo2019joint,
  title = {Joint Source-Target Self Attention with Locality Constraints},
  author = {Anonymous ACL submission 1573},
  booktitle = {Annual Meeting of the Association for Computational Linguistics},
  year = {2019},
  url = {http://www.acl2019.org/EN/index.xhtml},
}
```

## Translation

### Pre-trained models
Dataset | Model | Test set
---|---|---
[IWSLT14 German-English](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | [download (.tar.bz2)](https:// /iwslt14.de-en.local.tar.bz2) | IWSLT14 test: <br> [download (.tar.bz2)](https:// /data/iwslt14.31K.de-en.test.tar.bz2)
[WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https:// /wmt16.en-de.joined-dict.local.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
LightConv | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https:// /wmt14.en-fr.joined-dict.local.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https:// /data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)

### IWSLT14 De-En
Pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf) with a joint BPE with 31K tokens
```sh
# Dataset download and preparation
cd examples
#./prepare-iwslt14-31K.sh
cd ..

# Dataset binarization:
TEXT=examples/iwslt14.tokenized.31K.de-en
fairseq-preprocess --joined-dictionary --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.joined-dictionary.31K.de-en
```

Training and evaluating Local Joint Attention on a GPU:
```sh
# Training
SAVE="checkpoints/local_joint_attention_iwslt_de_en"
mkdir -p $SAVE

fairseq-train data-bin/iwslt14.joined-dictionary.31K.de-en \
    --user-dir models \
    --arch local_joint_attention_iwslt_de_en \
    --clip-norm 0 --optimizer adam --lr 0.001 \
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
fairseq-generate data-bin/iwslt14.joined-dictionary.31K.de-en --user-dir models --path "${SAVE}/checkpoint_last10_avg.pt" \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 1.7 --gen-subset test --quiet
```

### WMT16 En-De
Training and evaluating Local Joint Attention on WMT16 En-De using cosine scheduler on one machine with 8 V100 GPUs:
```sh
# Training
SAVE="save/joint_attention_wmt_en_de_big"
mkdir -p $SAVE
python -m torch.distributed.launch --nproc_per_node 8 fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --user-dir models \
    --arch local_joint_attention_wmt_en_de_big \
    --fp16 --log-interval 100 --no-progress-bar \
    --max-update 30000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 32 --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 1800 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.0009 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 20000 \
    --save-dir $SAVE

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt16.en-de.joined-dict.newstest2014 --user-dir models \
    --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 5 --remove-bpe --lenpen 0.35 --gen-subset test > wmt16_gen.txt
bash scripts/compound_split_bleu.sh wmt16_gen.txt
```

### WMT14 En-Fr
Training and evaluating Local Joint Attention on WMT14 En-Fr using cosine scheduler on one machine with 8 V100 GPUs:
```sh
# Training
SAVE="save/dynamic_conv_wmt14en2fr"
mkdir -p $SAVE
python -m torch.distributed.launch --nproc_per_node 8 fairseq-train \
    data-bin/wmt14_en_fr \
    --user-dir models \
    --arch local_joint_attention_wmt_en_fr_big \
    --fp16 --log-interval 100 --no-progress-bar \
    --max-update 80000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 32 --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 1800 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.0005 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 70000 \
    ---save-dir $SAVE

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14_en_fr --user-dir models \
    --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 5 --remove-bpe --lenpen 0.9 --gen-subset test
```

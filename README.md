# Joint Source-Target Self Attention with Locality Constraints (Fonollosa et al., 2019)
<<<<<<< HEAD
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
=======
This repository contains the source code, pre-trained models, as well as instructions to reproduce the results or [our paper](http://arxiv.org/abs/1905.06596)

## Citation:
```bibtex
@article{fonollosa2019joint,
  title={Joint Source-Target Self Attention with Locality Constraints},
  author={Jos\'e A. R. Fonollosa and Noe Casas and Marta R. Costa-juss\`a},
  journal={arXiv preprint arXiv:1905.06596},
  url={http://arxiv.org/abs/1905.06596}
  year={2019}
}
```

## Setup

### Requirements

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* [fairseq](https://github.com/pytorch/fairseq) version >= 0.6.2
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

### Install fairseq from source
```sh
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
cd ..
```

### Clone this repository
```sh
git clone https://github.com/jarfo/joint.git
cd joint
```

## Translation

### Pre-trained models
Dataset | Model | Prepared test set
---|---|---
[IWSLT14 German-English](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | [download (.pt)](https://github.com/jarfo/joint/releases/download/v0.1/local_joint_attention_iwslt_de_en.pt) | IWSLT14 test: [download (.tgz)](https://github.com/jarfo/joint/releases/download/v0.1/iwslt14.31K.de-en.test.tgz)
[WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.bz2)](https://github.com/jarfo/joint/releases/download/v0.1/local_joint_attention_wmt_en_de_big.pt.bz2) | newstest2014 (shared vocab): [download (.tgz)](https://github.com/jarfo/joint/releases/download/v0.1/wmt16_en_de_bpe32k.test.tgz)
[WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (split #1)](https://github.com/jarfo/joint/releases/download/v0.1/local_joint_attention_wmt_en_fr_big.pt_aa) <br> [download (split #2)](https://github.com/jarfo/joint/releases/download/v0.1/local_joint_attention_wmt_en_fr_big.pt_ab) | newstest2014 (shared vocab): [download (.tgz)](https://github.com/jarfo/joint/releases/download/v0.1/wmt14_en_fr.test.tgz)

The English-French model download is split in two files that can be joined with:
```sh
cat local_joint_attention_wmt_en_fr_big.pt_* > local_joint_attention_wmt_en_fr_big.pt
```

### **IWSLT14 De-En**
The IWSLT'14 German to English translation database ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf) is tokenized with a joint BPE with 31K tokens
```sh
# Dataset download and preparation
cd examples
./prepare-iwslt14-31K.sh
>>>>>>> 901e4684039b0da1c4c06b2c9fa1db49753daf50
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
<<<<<<< HEAD
fairseq-generate data-bin/iwslt14.joined-dictionary.31K.de-en --user-dir models --path "${SAVE}/checkpoint_last10_avg.pt" \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 1.7 --gen-subset test --quiet
```

### WMT16 En-De
Training and evaluating Local Joint Attention on WMT16 En-De using cosine scheduler on one machine with 8 V100 GPUs:
=======
fairseq-generate data-bin/iwslt14.joined-dictionary.31K.de-en --user-dir models \
    --path "${SAVE}/checkpoint_last10_avg.pt" \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 1.7 --gen-subset test --quiet
```

### **WMT16 En-De**
Training Local Joint Attention on WMT16 En-De using cosine scheduler on one machine with 8 Nvidia V100-16GB GPUs:

Download the [preprocessed WMT'16 En-De data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8). Then 

Extract the WMT'16 En-De data:
```sh
$ TEXT=wmt16_en_de_bpe32k
$ mkdir $TEXT
$ tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```

Preprocess the dataset with a joined dictionary:
```sh
$ fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary
```

Train a model
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

# Checkpoint averaging
python ../fairseq/scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# Evaluation on newstest2014
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --user-dir models \
    --path "${SAVE}/checkpoint_last10_avg.pt" \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 0.35 --gen-subset test > wmt16_gen.txt
bash ../fairseq/scripts/compound_split_bleu.sh wmt16_gen.txt
```

### **WMT14 En-Fr**
Training and evaluating Local Joint Attention on WMT14 En-Fr using cosine scheduler on one machine with 8 Nvidia V100-16GB GPUs:
```sh
# Data preparation
$ cd examples
$ bash prepare-wmt14en2fr.sh
$ cd ..

# Binarize the dataset:
$ TEXT=examples/wmt14_en_fr
$ fairseq-preprocess --joined-dictionary --source-lang en --target-lang fr \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0

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

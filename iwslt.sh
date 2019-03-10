# Dataset download and preparation
cd examples
#./prepare-iwslt14-31K.sh
cd ..

# Binarize the dataset:
TEXT=examples/iwslt14.tokenized.31K.de-en
fairseq-preprocess --joined-dictionary --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.joined-dictionary.31K.de-en

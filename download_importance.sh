#!/bin/bash

ENCODER_DIR=encoder
VECTOR_DIR=vectors

# download infersent models
mkdir -p $ENCODER_DIR
curl -Lo $ENCODER_DIR/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
curl -Lo $ENCODER_DIR/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

# download the word vectors that you need
# mkdir -p $VECTOR_DIR
# curl -Lo $VECTOR_DIR/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
# unzip $VECTOR_DIR/glove.840B.300d.zip -d $VECTOR_DIR/
# curl -Lo $VECTOR_DIR/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
# unzip $VECTOR_DIR/crawl-300d-2M.vec.zip -d $VECTOR_DIR/
#!/bin/sh
set -eux

TARGET_DIR=/data
echo ${TARGET_DIR}
mkdir -p ${TARGET_DIR}

cd ${TARGET_DIR}

# データセットをKaggle APIでダウンロードし、${TARGET_DIR}/input/以下に保存する
INPUT_DIR=input
if [ -e ${INPUT_DIR} ]; then
  echo ${INPUT_DIR} already exists, and we avoid downloading competition files
else
  mkdir ${INPUT_DIR}
  cd ${INPUT_DIR}
  echo "Download competition files"
  kaggle competitions download quora-question-pairs
  unzip quora-question-pairs.zip
  unzip train.csv.zip
  rm *.zip
  cd ..
fi

# 訓練済み埋め込みベクトルをダウンロードし、${TARGET_DIR}/embeddings/以下に保存する
EMBEDDINGS_DIR=embeddings
if [ -e ${EMBEDDINGS_DIR} ]; then
  echo ${EMBEDDINGS_DIR} already exists, and we avoid downloading competition files
else
  mkdir ${EMBEDDINGS_DIR}
  cd ${EMBEDDINGS_DIR}

  echo "Download pretrained GloVe embeddings"
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  echo "Convert the GloVe embeddings into binary format"
  python -c "from texts.embeddings2bin import embedding2bin; embedding2bin('glove.840B.300d.bin', 'glove.840B.300d.txt', 'glove.840B.300d.zip', True)"

  cd ..
fi

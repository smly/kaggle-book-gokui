# 第4章 画像検索入門

## ファイルの説明

## 準備

以下を確認して下さい。

* NVIDIA Driver, Docker, NVIDIA Container Toolkit 等が正しく設定されており、Docker コンテナ内で GPU が利用可能である。
* `${HOME}/.kaggle/kaggle.json` が正しく設定されており、Kaggle CLI が正しく動作する。

実行環境は Docker で実行できるように整備しています。
このディレクトリから以下のコマンドを実行することで Docker イメージをビルドできます。

```sh
$ docker build -t kaggle-book-ch4 .
```

## データの取得

必要となるデータは Kaggle コマンドからダウンロードすることができます。

```
$ kaggle datasets download -d qiubit/roxfordparis -p data
$ kaggle datasets download -d confirm/google-landmark-dataset-v2-micro -p data
$ unzip data/roxfordparis.zip -d data
$ unzip google-landmark-dataset-v2-micro -d data
```

## コードの実行

コードは Docker コンテナから実行する想定で用意しています。用意したデータを Docker コンテナ内で利用できるよう、コンテナ起動時に以下のようにマウントしてください。

| データセット名 | マウント元 | マウント先 |
|:----|:----|:----|
| ROxford5k | `data/roxford5k` | `/workspace/roxford5k` |
| RParis6k | `data/roxford5k` | `/workspace/rparis6k` |
| GLDv2-Micro | `data/google-landmark-dataset-v2-micro` | `:/workspace/gldv2_micro` |

実行例は以下のようになります。GPU が使える環境で実行する必要があります。

```sh
$ docker run --gpus all -it --rm --shm-size 2G \
  -v $HOME/.kaggle:/root/.kaggle \
  -v $HOME/.cache:/root/.cache \
  -v `pwd`/data/roxford5k:/workspace/roxford5k \
  -v `pwd`/data/rparis6k:/workspace/rparis6k \
  -v `pwd`/data/gldv2_micro:/workspace/gldv2_micro \
  -v `pwd`/out:/workspace/out \
  kaggle-book-ch4 \
  bash
```

添付コードはすべて `/workspace` にコピーされています。コンテナから以下のように実行することができます。

```sh
root@80c6595c745a:/workspace# python 01_baseline_off-the-shelf_gem.py
root@80c6595c745a:/workspace# python 02_baseline_benchmark.py
root@80c6595c745a:/workspace# python 03_baseline_random.py
root@80c6595c745a:/workspace# python 05_class_imbalance.py
root@80c6595c745a:/workspace# python 06_split_train_val_dataset.py
root@80c6595c745a:/workspace# python 07_gldv2_arcface_finetuning.py
root@80c6595c745a:/workspace# python 08_extract_local_features.py
root@80c6595c745a:/workspace# python 09_ransac_matching_local_features.py
root@80c6595c745a:/workspace# python 10_reranking_with_matching.py
root@80c6595c745a:/workspace# python 11_reranking_with_dba.py
root@80c6595c745a:/workspace# python 12_kaggle_gld21_benchmark_training.py
```

以下のスクリプトは Kaggle コンテスト「[Google Landmark Retrieval Challenge 2021](https://www.kaggle.com/competitions/landmark-retrieval-2021/overview)」に投稿する想定で用意しています。
`out/` に出力された学習済みモデルの重みファイルを Kaggle データセットとしてアップロードして Kaggle notebook 上でマウントして使用してください。

| ファイル名 | Kaggle Code |
|:----|:----|
| 13_kaggle_gld21_inference.py | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/confirm/13-kaggle-gld21-inference-py/notebook) |

同じ要領で [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark) のような大規模なデータセットを用いて学習すると更にスコアを上げることができます。大規模なデータセットを用いて効率的に試行錯誤するためには様々な技術あります。Automatic Mixed Precision (AMP) による混合精度学習、複数GPUを用いた分散学習、TPU による学習など、効率的に実験をしてスコア改善を目指してみましょう。

以下は GLDv2 clean によって学習したモデルを使ったノートブックの例です:

| ファイル名 | Kaggle Code |
|:----|:----|
| 13_kaggle_gld21_inference_gldv2clean.py | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/confirm/13-kaggle-gld21-inference-gldv2clean-py/notebook) |

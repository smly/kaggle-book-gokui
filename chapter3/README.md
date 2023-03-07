# 第3章 ソースコード

## ファイルの説明
 
* `train.py` --- 本章の実験で実際に用いたソースコード
* `run_all.py` --- 各節の設定に対応した学習・評価を行っていくための補助スクリプト
* `Dockerfile`, `requirements.txt` --- 実験で用いた環境を再現するためのファイル
* `run.sh` --- Docker コンテナ内で実験を行うための補助スクリプト


## 準備

以下を確認して下さい。

* NVIDIA Driver, Docker, NVIDIA Container Toolkit 等が正しく設定されており、Docker コンテナ内で GPU が利用可能である。
* `${HOME}/.kaggle/kaggle.json` が正しく設定されており、Kaggle CLI が正しく動作する。

また、データセットを書籍 p.64 「3.4.1 データセットの準備」に示すディレクトリ構造になるように配置して下さい。以下はそのためのコマンドの例です。

```
unzip -q -n ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d dogs_vs_cats
unzip -q -n ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d dogs_vs_cats
mkdir -p dogs_vs_cats/train/cat
mkdir -p dogs_vs_cats/train/dog
mkdir -p dogs_vs_cats/test/unknown
mv dogs_vs_cats/train/cat*.jpg dogs_vs_cats/train/cat
mv dogs_vs_cats/train/dog*.jpg dogs_vs_cats/train/dog
mv dogs_vs_cats/test/*.jpg dogs_vs_cats/test/unknown
```

## 実行

`run.sh` は、`Dockerfile` のイメージをビルドし、引数に与えられたコマンドをコンテナ内で実行するようになっています。従って、以下のようにすることで、本章の全ての実験を実行することができます。


```
DATA_DIR=<データを置いたディレクトリ> ./run.sh python ./run_all.py --data_dir=/data
```

データを置いているディレクトリは環境変数 `DATA_DIR` として与えて下さい。データは、Kaggleのサイトでダウンロードした後、本章で説明した通りにディレクトリ構造を変更して下さい。`run.sh` 内の設定により、コンテナ内では `DATA_DIR` は `/data` にマウントするようにしていますので、 `--data_dir=/data` の引数を付与して下さい。

## 実行結果

著者が実行した際の出力を `out/runall_2021-12-04_14-45-58` に置いてあります。

書籍でも説明した通り、疑似乱数を用いているため、再度実行しても、全く同じ結果は得られませんが、概ね傾向は一致するはずです。

### Validation

```
0      split   0.061287      5
1      split   0.036419      6
2      split   0.019586    7-1
3      split   0.016292    7-3
4         CV   0.019308    8-1
5         CV   0.018261    8-2
6         CV   0.012941    9-1
```

### LB

```
fileName      date                 description                                            status    publicScore  privateScore  
------------  -------------------  -----------------------------------------------------  --------  -----------  ------------  
out_clip.csv  2021-12-05 00:44:52  out/runall_2021-12-04_14-45-58/config9-1/out_clip.csv  complete  0.03868      0.03868       
out_clip.csv  2021-12-04 20:22:25  out/runall_2021-12-04_14-45-58/config8-2/out_clip.csv  complete  0.04276      0.04276       
out_clip.csv  2021-12-04 18:16:14  out/runall_2021-12-04_14-45-58/config8-1/out_clip.csv  complete  0.04294      0.04294       
out_clip.csv  2021-12-04 16:11:49  out/runall_2021-12-04_14-45-58/config7-3/out_clip.csv  complete  0.04453      0.04453       
out_clip.csv  2021-12-04 15:46:49  out/runall_2021-12-04_14-45-58/config7-1/out_clip.csv  complete  0.04660      0.04660       
out_clip.csv  2021-12-04 15:17:57  out/runall_2021-12-04_14-45-58/config6/out_clip.csv    complete  0.05441      0.05441       
out_clip.csv  2021-12-04 14:49:06  out/runall_2021-12-04_14-45-58/config5/out_clip.csv    complete  0.08274      0.08274           
```

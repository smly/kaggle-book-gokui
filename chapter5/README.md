# 第5章 ソースコード

## ファイルの説明


## 準備

以下を確認して下さい。

* NVIDIA Driver, Docker, NVIDIA Container Toolkit 等が正しく設定されており、Docker コンテナ内で GPU が利用可能である。
* `${HOME}/.kaggle/kaggle.json` が正しく設定されており、Kaggle CLI が正しく動作する。


## データの取得
`run.sh`は、第3章で使用した`run.sh`とほぼ同じスクリプトで、Dockerfile のイメージをビルドし、引数に与えられたコマンドをコンテナ内で実行するようになっています。第5章では、LightGBMやニューラルネットワークの実験を始める前に、以下のように`setup.sh`を実行します。`setup.sh`の中では、Kaggle APIを用いて訓練セット`train.csv`とテストセット`test.csv`をダウンロードして保存し、GloVe訓練済み単語ベクトルをダウンロードしてバイナリ形式で保存します。
```
DATA_DIR=<データを保存したいディレクトリ> ./run.sh ./setup.sh
```

例えば、著者の環境で`DATA_DIR=~/data  ./run.sh ./setup.sh`を実行すると、以下のようにデータが保存されます。
<pre>
/home/kaggle/data
├── embeddings
│   ├── glove.840B.300d.bin
│   └── glove.840B.300d.zip
└── input
    ├── test.csv
    └── train.csv
</pre>


## 実行
### LightGBM
LightGBMの実行例は以下のようになります。`NUM_PROCESSES`はLightGBMの並列計算に使用するコア数などに対応する変数なので、実行環境に合わせて設定して下さい。`DATA_DIR`は、データの取得時に`DATA_DIR`として指定した値と同じものを指定して下さい。以降の説明では`DATA_DIR=~/data`とします。
```NUM_PROCESSES=4 DATA_DIR=~/data ./run.sh python experiments/000_gbm_match.py```

実行結果として下記のようなファイルが出力されます。`booster.0.txt`や`booster.1.txt`は、0番目や1番目のfoldを検証セットとする訓練から得られたLightGBMモデルに対応しています。また、`oof_prediction.csv`は訓練セットのout-of-foldのデータへの予測が保存され、`tst_prediction.csv`は`booster.0.txt`から`booster.4.txt`までのモデルのテストセットへの予測を平均したものが保存されています。
<pre>
data/output/000_gbm_match/
├── booster.0.txt
├── booster.1.txt
├── booster.2.txt
├── booster.3.txt
├── booster.4.txt
├── oof_prediction.csv
├── score.txt
└── tst_prediction.csv
</pre>


### RNN & トランスフォーマー
RNNとトランスフォーマーの実験の実行方法もLightGBMとほぼ同じで、以下のように実行することで、各foldに対応するモデルとouf-of-foldへの予測とテストセットへの予測をそれぞれ計算することができます。
```DATA_DIR=~/data ./run.sh python experiments/204_bert_concat_last4cls.py --needs_predict --fold_all```

LightGBMの時と同様に、`oof_prediction.csv`は訓練セットのout-of-foldのデータへの予測が保存され、`tst_prediction.csv`は`booster.0.txt`から`booster.4.txt`までのモデルのテストセットへの予測を平均したものが保存されています。また、`summary.csv`には検証セットにおけるlossや実行時間などが保存されています。

**注意**: トランスフォーマーに基づくモデルの学習には多くの時間が必要となるため、`experiments`以下のスクリプトをすべて実行しようとすると、V100上でも数十時間が必要になります。そのため、一部の実験スクリプトのみを実行するか、あるいは`--dryrun`オプションを指定し一部のデータのみを使った動作確認だけを行うかのどちらかをおすすめします。

<pre>
data/output/204_bert_concat_last4cls/
├── config.json
├── model_0.bin
├── model_1.bin
├── model_2.bin
├── model_3.bin
├── model_4.bin
├── oof_prediction.csv
├── oof_prediction_0.csv
├── oof_prediction_1.csv
├── oof_prediction_2.csv
├── oof_prediction_3.csv
├── oof_prediction_4.csv
├── result_0.json
├── result_1.json
├── result_2.json
├── result_3.json
├── result_4.json
├── summary.csv
├── tst_prediction.csv
├── tst_prediction_0.csv
├── tst_prediction_1.csv
├── tst_prediction_2.csv
├── tst_prediction_3.csv
└── tst_prediction_4.csv
</pre>

### 予測の提出
`run.sh`を使ってDockerコンテナ内でKaggle APIを使用することで、予測結果を提出することができます。以下の例では、 最初に訓練したLightGBMによる予測をKaggle APIを用いて提出しています。
```
 DATA_DIR=~/qqp_data ./run.sh kaggle competitions submit -f /data/output/000_gbm_match/tst_prediction.csv -m 000_gbm_match  quora-question-pairs
 ```
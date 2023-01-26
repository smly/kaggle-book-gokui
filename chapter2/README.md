# 第2章 探索的データ分析とモデルの作成・検証・性能向上

## 各ディレクトリ・ファイルの内容

| ディレクトリ・ファイル名 | 説明 | 本文の登場箇所 |
|:---|:---|:---|
| out/ | 生成したファイルを保存 | - |
| 00_mlp.py | 3層のパーセプトロンの学習 | 2.2.1　ニューラルネットワーク |
| 01_mlp_batch.py | ミニバッチの利用 | 2.2.1　ニューラルネットワーク |
| 02_mlp_hold_out.py | ホールドアウト検証 | 2.3.2　検証セットが必要な理由 |
| 03_mlp_kfold.py | k-fold交差検証 | 2.3.3　交差検証 |
| 04_mlp_stratified_kfold.py | stratified k-fold交差検証 | 2.3.4　多様な検証方法の使い分け |
| Dockerfile | Dockerで作成するコンテナイメージの管理 | - |
| requirements.txt | Pythonライブラリのバージョンの指定 | - |
| setup.sh | 環境構築のための実行ファイル | - |

## 実行コマンド

```bash
sh setup.sh
python {file_name.py}
```

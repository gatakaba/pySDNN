pySDNN
===============================================

## 概要
- PP・MLP・SDNNのpythonによる実装

## 特徴
- scikit-learnのフォーマットで学習・実行が可能
- 純python製
- docker内のjupyter notebookを用いてブラウザ上で動作可能


## インストール方法
予めpython(3.×)がインストールされていることを確認して下さい.

依存ライブラリのインストール中にエラーが生じることがあります.  
その場合は場合は主要ライブラリをオールインワンでインストールすることができる
[anaconda](https://www.continuum.io/)
を使うことにより回避することができる場合があります.


### 1. ダウンロード
`$ git clone https://github.com/gatakaba/pySDNN`

### 2. ライブラリが存在するディレクトリに移動
`$ cd /path/to/pySDNN` 

### 3. 依存ライブラリをインストール
`$ pip install -r requirements.txt`

### 4. ライブラリをインストール
`$ python setup.py install`


## jupyter-notebookからSDNNを使う
仮想環境構築ソフトdockerを用いることによって,インストールすることなく,
SDNNを使うことができます.


### 1. 準備

[docker](https://docs.docker.com/engine/installation/)をインストール


### 2. Docker imageのダウンロード

`$ docker pull gatakaba/pySDNN`



### 3. Dockerコンテナの起動

`$  docker run -it sdnn /bin/bash`


#### 4. サーバに接続

ブラウザを使って`localhost:5000`にアクセス


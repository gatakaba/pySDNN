pySDNN
===============================================

## 概要
- PP・MLP・SDNNのpythonによる実装

## 特徴
- scikit-learn形式でモデルの学習・実行が可能
- 純python製
- docker内のjupyter notebookを用いることによってブラウザ上で動作可能


## インストール方法
予めpython(3.×)がインストールされていることを確認して下さい.

依存ライブラリのインストール中にエラーが生じることがあります.  
その場合はpythonを[anaconda](https://www.continuum.io/)を用いて
再インストールすることにより回避することができます.


### 1. ダウンロード
`$ git clone https://github.com/gatakaba/pySDNN`

### 2. ライブラリが存在するディレクトリに移動
`$ cd /path/to/pySDNN` 

### 3. 依存ライブラリをインストール
`$ pip install -r requirements.txt`

### 4. ライブラリをインストール
`$ python setup.py install`


## ブラウザからSDNNを使う
仮想環境構築ソフトdockerを用いることによって本パッケージをインストールすることなく,SDNNを使うことができます.

### 1. Dockerのインストール

[docker](https://docs.docker.com/engine/installation/)をインストール


### 2. Docker イメージのダウンロード

`$ docker pull gatakaba/pySDNN`



### 3. Dockerコンテナの起動

`$  docker run -d -p 5000:8888 pysdnn jupyter-notebook`


#### 4. サーバに接続

ブラウザのアドレスバーに`localhost:5000`を打ち込み,jupyter notebookにアクセス


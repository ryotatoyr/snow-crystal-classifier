# 機械学習を用いた粒子（霰・雪片）の自動分類

雪の結晶画像から「霰（あられ）」と「雪片」を自動で見分ける機械学習プログラムです。OpenCVで画像から特徴量を抽出し、RandomForestで分類します。

## セットアップ

### 必要な環境

- git（バージョン管理システム）
- uv（パッケージマネージャ）

※ gitがインストールされていない人は下記を参照にインストールしてください。**手順2まででOK**です。

https://qiita.com/takeru-hirai/items/4fbe6593d42f9a844b1c


※ uvがインストールされていない人は下記コマンドでインストールしてください。

```bash
pip install uv

# macOSの場合は下記を推奨
brew install uv
```


### インストール

```bash
git clone https://github.com/matsuda-tkm/snow-crystal-classifier.git
cd snow-crystal-classifier
uv sync
```

### データセットの準備

```bash
unzip dataset.zip
```

これで `dataset/` フォルダに画像データが展開されます。

## 使い方

### 分類器の実行

```bash
uv run scripts/main.py
```

実行すると、クロスバリデーションで評価を行い、結果を `output/` フォルダに保存します。

オプション:

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| --data-dir | データセットのパス | dataset |
| --output-dir | 出力先のパス | output |
| --n-folds | クロスバリデーションの分割数 | 5 |
| --seed | 乱数シード | 42 |

### 決定木の可視化

```bash
uv run scripts/visualize_tree.py
```

実行すると、以下のファイルが生成されます:

| ファイル | 内容 |
|---------|------|
| decision_tree.png | 決定木の構造図 |
| feature_importance.png | 特徴量の重要度ランキング |

## ファイル構成

```
snow-crystal-classifier/
├── scripts/
│   ├── main.py              # メインスクリプト（クロスバリデーション評価）
│   └── visualize_tree.py    # 決定木の可視化
├── src/
│   └── snow_crystal_classifier/
│       ├── __init__.py
│       ├── classifier.py    # 分類器の実装
│       └── utils.py         # データ読み込み
├── dataset/                 # データセット
│   ├── graupel/            # 霰の画像
│   └── snowflake/          # 雪片の画像
├── output/                  # 出力結果
├── pyproject.toml          # 依存関係の定義
└── README.md
```

## データセット

`dataset/` フォルダに以下の構成で画像を配置してください。

```
dataset/
├── graupel/      # 霰の画像 (107枚)
│   ├── image1.png
│   └── ...
└── snowflake/    # 雪片の画像 (323枚)
    ├── image1.png
    └── ...
```

## 出力ファイル

実行後、`output/` フォルダに以下のファイルが生成されます。

| ファイル | 内容 |
|---------|------|
| confusion_matrix.png | 混同行列（予測と正解の対応表） |

## 課題内容

### 課題1: クロスバリデーションの実装

`scripts/main.py` の `run_cross_validation` 関数を完成させてください。

現在、関数の一部がコメントアウトされています。コメントのヒントを参考に、以下を実装してください：

- 各Foldでの訓練と評価
- 評価指標の計算
- 全Foldの結果の集計

実装が完了すると、プログラムが正常に動作するようになります。

### 課題2: 特徴量抽出の実装

`src/snow_crystal_classifier/classifier.py` の `_extract_features` メソッドに特徴量を追加してください。

現在は円形度のみが実装されています。これを参考に、独自の特徴量を実装してみましょう。

特徴量の例：
- 形状特徴: 面積比、アスペクト比、複雑さ など
- テクスチャ特徴: LBP（Local Binary Pattern）、Gaborフィルタ など
- エッジ特徴: Canny、Sobel、Laplacian など
- 統計特徴: 明るさの分布、エントロピー など

参考:
- OpenCV公式ドキュメント: https://docs.opencv.org/4.x/
- 画像処理チュートリアル: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html


### 発展課題: 精度向上への挑戦

さらに精度を上げるために、以下に挑戦してみましょう：

#### パラメータチューニング

- `src/snow_crystal_classifier/classifier.py` の `RandomForestClassifier` のパラメータを調整
  - `n_estimators`: 決定木の数
  - `max_depth`: 決定木の最大深さ
  
- `scripts/visualize_tree.py` の `DecisionTreeClassifier` の `max_depth` を変更して可視化

#### 誤分類パターンの分析

- どのような画像が誤って分類されているか調べる
- 誤分類される画像の特徴を分析する
- 分析結果をもとに特徴量を改良する

## 使用ライブラリ

- OpenCV: 画像処理と特徴量抽出
- scikit-learn: RandomForest分類器と評価
- NumPy: 数値計算
- Matplotlib / Seaborn: 可視化

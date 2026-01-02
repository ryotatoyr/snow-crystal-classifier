"""
OpenCV特徴量ベース雪晶分類器のクロスバリデーション評価

霰(graupel)と雪片(snowflake)をRandomForestで分類し、
クロスバリデーションで評価する
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snow_crystal_classifier import SnowCrystalClassifier



def resize_with_padding(image, target_size):
    """アスペクト比を維持しながらパディングでサイズを揃える"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    pad_h, pad_w = (target_h - new_h) // 2, (target_w - new_w) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded


def load_dataset(data_dir, image_size):
    """データセットを読み込む"""
    class_names = ["graupel", "snowflake"]
    images, labels = [], []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for img_path in sorted(class_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(resize_with_padding(img, image_size))
                labels.append(label)

    return np.array(images), np.array(labels), class_names


def compute_metrics(y_true, y_pred):
    """
    評価メトリクスを計算する

    戻り値:
        metrics: 評価指標を格納した辞書
            - "accuracy": 正解率
            - "precision": 適合率
            - "recall": 再現率
            - "f1": F1スコア
            - "confusion_matrix": 混同行列
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def run_cross_validation(X, y, n_folds=5, random_seed=42):
    """
    クロスバリデーションを実行する

    Args:
        X: 画像データ (N, H, W, C)
        y: ラベル (N,)
        n_folds: 分割数
        random_seed: 乱数シード

    処理の流れ:
        1. StratifiedKFoldでデータをn_folds個に分割
        2. 各Foldで: 訓練データで学習 → テストデータで予測 → 評価
        3. 全Foldの結果を集計

    戻り値:
        mean_metrics: 平均評価指標 (dict)
        std_metrics: 標準偏差 (dict)
    """
    # StratifiedKFoldでデータを分割（クラス比率を維持）
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # 各Foldの評価結果を保存するリスト
    fold_metrics = []
    # 全Foldの予測結果を保存（混同行列用）
    all_preds = []
    all_labels = []

    # 各Foldで訓練と評価を実行
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # 訓練データとテストデータに分割
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 分類器を作成して訓練
        clf = SnowCrystalClassifier(random_state=random_seed)
        clf.fit(X_train, y_train)

        # テストデータで予測
        y_pred = clf.predict(X_test)

        # 評価指標を計算
        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

        # 全予測結果を保存
        all_preds.extend(y_pred)
        all_labels.extend(y_test)

        print(f"  Fold {fold_idx + 1}/{n_folds} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # 平均と標準偏差を計算
    mean_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
        "precision": np.mean([m["precision"] for m in fold_metrics]),
        "recall": np.mean([m["recall"] for m in fold_metrics]),
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }
    std_metrics = {
        "accuracy": np.std([m["accuracy"] for m in fold_metrics]),
        "precision": np.std([m["precision"] for m in fold_metrics]),
        "recall": np.std([m["recall"] for m in fold_metrics]),
        "f1": np.std([m["f1"] for m in fold_metrics]),
    }

    return mean_metrics, std_metrics


def plot_confusion_matrix(metrics, class_names, output_path):
    """混同行列をプロットする"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"Confusion Matrix (Accuracy: {metrics['accuracy']:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metrics(metrics, std, output_path):
    """メトリクスをプロットする"""
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    errors = [std["accuracy"], std["precision"], std["recall"], std["f1"]]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, yerr=errors, capsize=5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("RandomForest Classifier Performance (5-Fold CV)")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_csv(metrics, std, output_path):
    """結果をCSVに保存する"""
    df = pd.DataFrame([{
        "accuracy_mean": metrics["accuracy"],
        "accuracy_std": std["accuracy"],
        "precision_mean": metrics["precision"],
        "precision_std": std["precision"],
        "recall_mean": metrics["recall"],
        "recall_std": std["recall"],
        "f1_mean": metrics["f1"],
        "f1_std": std["f1"],
    }])
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def main(data_dir, output_dir, n_folds, image_size, seed):
    """メイン関数"""
    print("=" * 60)
    print("OpenCV特徴量ベース雪晶分類器（RandomForest）")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("\nLoading data...")
    X, y, class_names = load_dataset(data_dir, (image_size, image_size))
    print(f"  Samples: {len(X)}, Classes: {dict(zip(class_names, np.bincount(y)))}")

    # クロスバリデーション
    print(f"\nRunning {n_folds}-fold cross validation...")
    metrics, std = run_cross_validation(X, y, n_folds, seed)

    # 結果表示
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} +/- {std['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} +/- {std['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f} +/- {std['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f} +/- {std['f1']:.4f}")

    # 保存
    print("\nSaving results...")
    plot_confusion_matrix(metrics, class_names, output_dir / "confusion_matrix.png")
    plot_metrics(metrics, std, output_dir / "metrics.png")
    save_results_csv(metrics, std, output_dir / "results.csv")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="雪晶分類器のクロスバリデーション評価")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir), args.n_folds, args.image_size, args.seed)

"""
OpenCV特徴量ベース雪晶分類器のクロスバリデーション評価

霰(graupel)と雪片(snowflake)をRandomForestで分類し、
クロスバリデーションで評価する
"""

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from snow_crystal_classifier import SnowCrystalClassifier
from snow_crystal_classifier.utils import load_dataset


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | np.ndarray]:
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


def run_cross_validation(
    images: list[np.ndarray],
    labels: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> tuple[dict[str, float | np.ndarray], dict[str, float]]:
    """
    クロスバリデーションを実行する

    Args:
        images: 画像データのリスト（各画像は(H, W, C)のRGB画像）
        labels: ラベル配列 (N,)
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
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
        # TODO: 以下を実装してください
        #
        # 1. train_idx と test_idx を使って、images と labels を訓練用とテスト用に分割
        train_images = [images[i] for i in train_idx]
        train_labels = labels[train_idx]
        test_images = [images[i] for i in test_idx]
        test_labels = labels[test_idx]

        # 2. SnowCrystalClassifier を作成して訓練
        classifier = SnowCrystalClassifier(random_state=random_seed)
        classifier.fit(train_images, train_labels)

        # 3. テストデータで予測
        test_preds = classifier.predict(test_images)

        # 4. compute_metrics 関数で評価指標を計算し、fold_metrics に追加
        metrics = compute_metrics(test_labels, test_preds)
        fold_metrics.append(metrics)
        # 5. 全予測結果を all_preds と all_labels に保存（後で混同行列を計算するため）
        all_preds.extend(test_preds)
        all_labels.extend(test_labels)
        # 6. 各Foldの進捗を表示（オプション）
        print(
            f"  Fold {fold_idx + 1}/{n_folds}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
        pass

    # TODO: 評価指標の平均と標準偏差を計算してください
    #
    mean_metrics: dict[str, float | np.ndarray] = {}
    std_metrics: dict[str, float] = {}
    # - fold_metrics から各指標の平均を計算して mean_metrics に格納
    mean_metrics = {
        "accuracy": (np.mean([m["accuracy"] for m in fold_metrics])),
        "precision": (np.mean([m["precision"] for m in fold_metrics])),
        "recall": (np.mean([m["recall"] for m in fold_metrics])),
        "f1": (np.mean([m["f1"] for m in fold_metrics])),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }
    # - fold_metrics から各指標の標準偏差を計算して std_metrics に格納
    std_metrics = {
        "accuracy": (np.std([m["accuracy"] for m in fold_metrics])),
        "precision": (np.std([m["precision"] for m in fold_metrics])),
        "recall": (np.std([m["recall"] for m in fold_metrics])),
        "f1": (np.std([m["f1"] for m in fold_metrics])),
    }
    # - 混同行列は all_labels と all_preds から計算
    # - mean_metrics と std_metrics を return

    return mean_metrics, std_metrics


def plot_confusion_matrix(
    metrics: dict[str, float | np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> None:
    """混同行列をプロットする"""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm: np.ndarray = cast(np.ndarray, metrics["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Accuracy: {metrics['accuracy']:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main(
    data_dir: Path,
    output_dir: Path,
    n_folds: int,
    seed: int,
) -> None:
    """メイン関数"""
    print("=" * 60)
    print("OpenCV特徴量ベース雪晶分類器（RandomForest）")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("\nLoading data...")
    images, labels, class_names = load_dataset(data_dir)
    class_counts = dict(zip(class_names, np.bincount(labels), strict=False))
    print(f"  Samples: {len(images)}, Classes: {class_counts}")

    # クロスバリデーション
    print(f"\nRunning {n_folds}-fold cross validation...")
    metrics, std = run_cross_validation(images, labels, n_folds, seed)

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

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="雪晶分類器のクロスバリデーション評価")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir), args.n_folds, args.seed)

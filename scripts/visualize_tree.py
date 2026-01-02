"""
決定木の可視化スクリプト

DecisionTreeClassifierを使用して霰/雪片を分類し、決定木を可視化する
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score

from snow_crystal_classifier import SnowCrystalClassifier
from snow_crystal_classifier.utils import load_dataset


def extract_features(images: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """
    画像から特徴量を抽出する
    
    Args:
        images: 画像データのリスト（各画像は(H, W, C)のRGB画像）
    
    Returns:
        特徴量配列と特徴量名のリスト
    """
    extractor = SnowCrystalClassifier()
    features = extractor.extract_features(images)
    feature_names = extractor.get_feature_names()
    return features, feature_names


def visualize_tree(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    output_path: Path,
) -> None:
    """決定木を可視化する"""
    fig, ax = plt.subplots(figsize=(24, 16))
    plot_tree(tree, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, fontsize=8, proportion=True)
    ax.set_title("Decision Tree", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_importance(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """特徴量の重要度を可視化する"""
    importances = tree.feature_importances_
    # 重要度の高い順にソート
    sorted_indices = np.argsort(importances)[::-1]  # 降順にソート
    # 上位top_n個を取得
    indices = sorted_indices[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap("viridis")
    # グラデーションカラーを生成（viridisカラーマップの0.2〜0.8の範囲を逆順で使用）
    colors = cmap(np.linspace(0.2, 0.8, top_n)[::-1])
    ax.barh(range(top_n), importances[indices][::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    print("決定木分類器の訓練と可視化")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("\nLoading data...")
    images, labels, class_names = load_dataset(data_dir)
    print(f"  Samples: {len(images)}")

    # 特徴量抽出
    print("Extracting features...")
    features, feature_names = extract_features(images)
    print(f"  Features: {features.shape[1]}")

    # クロスバリデーション
    tree = DecisionTreeClassifier(
        max_depth=7,
        class_weight="balanced",
        random_state=seed,
    )
    print("\nCross-validation...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(tree, features, labels, cv=cv)
    print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 全データで訓練
    print("\nTraining on full dataset...")
    tree.fit(features, labels)
    print(f"  Depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")

    # 可視化
    print("\nVisualizing...")
    visualize_tree(tree, feature_names, class_names, output_dir / "decision_tree.png")
    visualize_importance(tree, feature_names, output_dir / "feature_importance.png")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="決定木の訓練と可視化")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir), args.n_folds, args.seed)

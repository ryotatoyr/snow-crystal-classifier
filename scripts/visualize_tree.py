"""
決定木の可視化スクリプト

DecisionTreeClassifierを使用して霰/雪片を分類し、決定木を可視化する
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from snow_crystal_classifier import SnowCrystalClassifier


def resize_with_padding(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
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


def load_dataset(data_dir: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """データセットを読み込む"""
    class_names = ["graupel", "snowflake"]
    images, labels = [], []

    for label, class_name in enumerate(class_names):
        for img_path in sorted((data_dir / class_name).glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(resize_with_padding(img, image_size))
                labels.append(label)

    return np.array(images), np.array(labels), class_names


def extract_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    画像から特徴量を抽出する
    
    Returns:
        特徴量配列と特徴量名のリスト
    """
    extractor = SnowCrystalClassifier()
    features = extractor._extract_features_batch(images)
    feature_names = extractor.get_feature_names()
    return features, feature_names


def visualize_tree(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    output_path: Path,
    max_depth: int,
) -> None:
    """決定木を可視化する"""
    fig, ax = plt.subplots(figsize=(24, 16))
    plot_tree(tree, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, max_depth=max_depth, fontsize=8, proportion=True)
    ax.set_title(f"Decision Tree (depth={tree.get_depth()}, leaves={tree.get_n_leaves()})", fontsize=14)
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
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap("viridis")
    ax.barh(range(top_n), importances[indices][::-1], color=cmap(np.linspace(0.2, 0.8, top_n)[::-1]))
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
    max_depth: int,
    n_folds: int,
    image_size: int,
    seed: int,
) -> None:
    """メイン関数"""
    print("=" * 60)
    print("決定木分類器の訓練と可視化")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("\nLoading data...")
    X, y, class_names = load_dataset(data_dir, (image_size, image_size))
    print(f"  Samples: {len(X)}")

    # 特徴量抽出
    print("Extracting features...")
    features, feature_names = extract_features(X)
    print(f"  Features: {features.shape[1]}")

    # 標準化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # クロスバリデーション
    print(f"\nCross-validation (max_depth={max_depth})...")
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed)
    scores = cross_val_score(tree, features, y,
                             cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed))
    print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 全データで訓練
    print("\nTraining on full dataset...")
    tree.fit(features, y)
    print(f"  Depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")

    # 可視化
    print("\nVisualizing...")
    visualize_tree(tree, feature_names, class_names, output_dir / "decision_tree.png", max_depth)
    visualize_importance(tree, feature_names, output_dir / "feature_importance.png")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="決定木の訓練と可視化")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir), args.max_depth, args.n_folds, args.image_size, args.seed)

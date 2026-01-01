import argparse
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from classifier import SnowCrystalClassifier

console = Console()


@dataclass
class Metrics:
    """評価メトリクス"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


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
        class_dir = data_dir / class_name
        for img_path in sorted(class_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(resize_with_padding(img, image_size))
                labels.append(label)

    return np.array(images), np.array(labels), class_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """評価メトリクスを計算する"""
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),  # type: ignore[arg-type]
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),  # type: ignore[arg-type]
        f1=f1_score(y_true, y_pred, average="macro", zero_division=0),  # type: ignore[arg-type]
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> tuple[Metrics, dict]:
    """クロスバリデーションを実行する"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    all_preds, all_labels = [], []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Cross Validation", total=n_folds)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            clf = SnowCrystalClassifier(random_state=random_seed)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])

            metrics = compute_metrics(y[test_idx], y_pred)
            fold_metrics.append(metrics)
            all_preds.extend(y_pred)
            all_labels.extend(y[test_idx])

            progress.update(task, advance=1, description=f"[cyan]Fold {fold_idx + 1}/{n_folds} - Acc: {metrics.accuracy:.4f}, F1: {metrics.f1:.4f}")

    # 平均と標準偏差を計算
    mean_metrics = Metrics(
        accuracy=float(np.mean([m.accuracy for m in fold_metrics])),
        precision=float(np.mean([m.precision for m in fold_metrics])),
        recall=float(np.mean([m.recall for m in fold_metrics])),
        f1=float(np.mean([m.f1 for m in fold_metrics])),
        confusion_matrix=confusion_matrix(all_labels, all_preds),
    )
    std_metrics = {
        "accuracy": np.std([m.accuracy for m in fold_metrics]),
        "precision": np.std([m.precision for m in fold_metrics]),
        "recall": np.std([m.recall for m in fold_metrics]),
        "f1": np.std([m.f1 for m in fold_metrics]),
    }
    return mean_metrics, std_metrics


def plot_confusion_matrix(metrics: Metrics, class_names: list[str], output_path: Path) -> None:
    """混同行列をプロットする"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics.confusion_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"Confusion Matrix (Accuracy: {metrics.accuracy:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    console.print(f"  [green]✓[/green] Saved: [cyan]{output_path}[/cyan]")


def plot_metrics(metrics: Metrics, std: dict, output_path: Path) -> None:
    """メトリクスをプロットする"""
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1]
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
    console.print(f"  [green]✓[/green] Saved: [cyan]{output_path}[/cyan]")


def save_results_csv(metrics: Metrics, std: dict, output_path: Path) -> None:
    """結果をCSVに保存する"""
    df = pd.DataFrame([{
        "accuracy_mean": metrics.accuracy,
        "accuracy_std": std["accuracy"],
        "precision_mean": metrics.precision,
        "precision_std": std["precision"],
        "recall_mean": metrics.recall,
        "recall_std": std["recall"],
        "f1_mean": metrics.f1,
        "f1_std": std["f1"],
    }])
    df.to_csv(output_path, index=False)
    console.print(f"  [green]✓[/green] Saved: [cyan]{output_path}[/cyan]")


def main(data_dir: Path, output_dir: Path, n_folds: int, image_size: int, seed: int) -> None:
    """メイン関数"""
    console.print(Panel.fit(
        "[bold cyan]OpenCV特徴量ベース雪晶分類器（RandomForest）[/bold cyan]",
        border_style="cyan"
    ))

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    console.print("\n[bold yellow]Loading data...[/bold yellow]")
    X, y, class_names = load_dataset(data_dir, (image_size, image_size))
    console.print(f"  Samples: [green]{len(X)}[/green], Classes: {dict(zip(class_names, np.bincount(y)))}")

    # クロスバリデーション
    console.print(f"\n[bold yellow]Running {n_folds}-fold cross validation...[/bold yellow]")
    metrics, std = run_cross_validation(X, y, n_folds, seed)

    # 結果表示
    results_table = Table(title="Results", show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Mean", justify="right", style="green")
    results_table.add_column("Std", justify="right", style="yellow")

    results_table.add_row("Accuracy", f"{metrics.accuracy:.4f}", f"± {std['accuracy']:.4f}")
    results_table.add_row("Precision", f"{metrics.precision:.4f}", f"± {std['precision']:.4f}")
    results_table.add_row("Recall", f"{metrics.recall:.4f}", f"± {std['recall']:.4f}")
    results_table.add_row("F1", f"{metrics.f1:.4f}", f"± {std['f1']:.4f}")

    console.print()
    console.print(results_table)

    # 保存
    console.print("\n[bold yellow]Saving results...[/bold yellow]")
    plot_confusion_matrix(metrics, class_names, output_dir / "confusion_matrix.png")
    plot_metrics(metrics, std, output_dir / "metrics.png")
    save_results_csv(metrics, std, output_dir / "results.csv")

    console.print("\n[bold green]✓ Done![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="雪晶分類器のクロスバリデーション評価")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir), args.n_folds, args.image_size, args.seed)

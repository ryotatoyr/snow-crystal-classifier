"""共通ユーティリティ関数"""

from pathlib import Path

import cv2
import numpy as np


def load_dataset(data_dir: Path) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    データセットを読み込む
    
    Args:
        data_dir: データセットのディレクトリ
    
    Returns:
        images: 画像のリスト（各画像はサイズが異なる可能性あり）
        labels: ラベル配列
        class_names: クラス名のリスト
    """
    class_names = ["graupel", "snowflake"]
    images, labels = [], []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for img_path in sorted(class_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)

    return images, np.array(labels), class_names


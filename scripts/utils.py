"""共通ユーティリティ関数"""

from pathlib import Path

import cv2
import numpy as np


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


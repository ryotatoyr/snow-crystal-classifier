from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# from skimage.feature import local_binary_pattern


class SnowCrystalClassifier:
    """
    OpenCVを使用した画像処理特徴量に基づく雪晶分類器

    画像から特徴量を抽出し、RandomForestで霰(graupel)と雪片(snowflake)を分類する
    """

    def __init__(self, random_state=42):
        """
        Args:
            random_state: 乱数シード
        """
        self.classifier = RandomForestClassifier(
            n_estimators=100,  # 決定木の数
            max_depth=7,  # 決定木の最大深さ
            class_weight="balanced",  # クラスの重み付け
            random_state=random_state,  # 乱数シード
            n_jobs=-1,  # 並列処理の使用
        )
        self._feature_names = None

    def fit(self, X_train, y_train):
        """
        モデルを訓練する

        Args:
            X_train: 訓練画像データのリスト（各画像は(H, W, C)のRGB画像）
            y_train: 訓練ラベル (N,)

        Returns:
            self
        """
        features = self._extract_features_batch(X_train)
        self.classifier.fit(features, y_train)
        return self

    def predict(self, X):
        """
        予測を行う

        Args:
            X: 入力画像データのリスト（各画像は(H, W, C)のRGB画像）

        Returns:
            予測ラベル (N,)
        """
        features = self._extract_features_batch(X)
        return self.classifier.predict(features)

    def extract_features(self, images):
        """
        画像から特徴量を抽出する

        Args:
            images: 画像データのリスト（各画像は(H, W, C)のRGB画像）

        Returns:
            特徴量配列 (N, n_features)
        """
        return self._extract_features_batch(images)

    def _extract_features_batch(self, images):
        """複数画像から特徴量を抽出する"""
        return np.array([self._extract_features(img) for img in images])

    def get_feature_names(self):
        """
        特徴量名のリストを返す

        Returns:
            特徴量名のリスト
        """
        assert self._feature_names is not None
        return self._feature_names.copy()

    def _add_features(self, features, feature_names, feature_dict):
        """
        特徴量名と値を同時に追加する

        Args:
            features: 特徴量リスト
            feature_names: 特徴量名リスト（Noneの場合は名前を追加しない）
            feature_dict: 特徴量名と値の辞書
        """
        for name, value in feature_dict.items():
            if isinstance(value, (list, np.ndarray)):
                features.extend(value if isinstance(value, list) else value.tolist())
                if feature_names is not None:
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        # 0次元配列（スカラー）の場合
                        feature_names.append(name)
                    else:
                        # 配列の場合、各要素に名前を付ける
                        for i in range(len(value)):
                            feature_names.append(
                                f"{name}_{i}" if len(value) > 1 else name
                            )
            else:
                features.append(value)
                if feature_names is not None:
                    feature_names.append(name)

    def _extract_features(self, image):
        """
        1枚の画像から特徴量を抽出する

        Args:
            image: RGB画像 (H, W, 3)

        Returns:
            特徴量ベクトル (1次元配列)
        """
        # カラー画像をグレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        features = []
        # 特徴量名を同時に管理（初回のみ生成）
        if self._feature_names is None:
            feature_names = []
        else:
            feature_names = None  # 既に生成済みの場合は名前を追加しない

        # ========================================
        # 形状特徴量（実装例）
        # ========================================
        # 二値化して輪郭を抽出
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 初期値（輪郭なしの場合に備える）
        circularity = 0.0
        area_ratio = 0.0
        aspect_ratio = 0.0
        complexity = 0.0

        contours_list: list[np.ndarray] = list(cast(Sequence[np.ndarray], contours))
        if contours_list:

            def _contour_area(cnt: np.ndarray) -> float:
                return float(cv2.contourArea(cnt))

            # 最大の輪郭を取得
            largest = max(contours_list, key=_contour_area)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            # 円形度: 真円に近いほど1に近づく
            # 霰（丸い）は1に近く、雪片（複雑な形）は小さくなる
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

            # 面積比
            area_ratio = area / (gray.shape[0] * gray.shape[1])

            # アスペクト比
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h > 0 else 0

            # 複雑さ
            complexity = perimeter / np.sqrt(area) if area > 0 else 0

            # テクスチャ特徴
            # lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            # (lbp_hist, _) = np.histogram(
            #     lbp.ravel(),
            #     bins=np.arange(0, 10),  # uniform LBPのパターン数に基づく
            #     range=(0, 9),
            # )

        # 特徴量を追加
        self._add_features(
            features,
            feature_names,
            {
                "circularity": circularity,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "complexity": complexity,
                # "lbp_hist": lbp_hist / lbp_hist.sum(),
            },
        )

        # ========================================
        # TODO: ここに追加の特徴量を実装してください
        # ========================================
        # ヒント: 以下のような特徴量を追加してみましょう
        #
        # 形状特徴:
        #   - 面積比: area / (gray.shape[0] * gray.shape[1])
        #   - アスペクト比: バウンディングボックスの幅/高さ
        #   - 複雑さ: perimeter / np.sqrt(area)
        #
        # テクスチャ特徴:
        #   - LBP (Local Binary Pattern)
        #   - Gaborフィルタ
        #
        # エッジ特徴:
        #   - Cannyエッジ検出
        #   - Sobelフィルタ
        #
        # 統計特徴:
        #   - 明るさの平均・標準偏差
        #   - エントロピー
        #
        # 参考: OpenCV公式ドキュメント
        # https://docs.opencv.org/4.x/

        # 特徴量名をキャッシュ
        if feature_names is not None:
            self._feature_names = feature_names

        return np.array(features)

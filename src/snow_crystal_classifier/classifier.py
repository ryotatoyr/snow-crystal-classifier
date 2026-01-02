"""OpenCV特徴量ベースの雪晶分類器（RandomForest）"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class SnowCrystalClassifier:
    """
    OpenCVを使用した画像処理特徴量に基づく雪晶分類器

    画像から特徴量を抽出し、RandomForestで霰(graupel)と雪片(snowflake)を分類する
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Args:
            n_estimators: 決定木の数
            max_depth: 決定木の最大深さ
            random_state: 乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._feature_names = None  # 特徴量名をキャッシュ

    def fit(self, X_train, y_train):
        """
        モデルを訓練する

        Args:
            X_train: 訓練画像データ (N, H, W, C)
            y_train: 訓練ラベル (N,)

        Returns:
            self
        """
        features = self._extract_features_batch(X_train)
        features = self.scaler.fit_transform(features)
        self.classifier.fit(features, y_train)
        return self

    def predict(self, X):
        """
        予測を行う

        Args:
            X: 入力画像データ (N, H, W, C)

        Returns:
            予測ラベル (N,)
        """
        features = self._extract_features_batch(X)
        features = self.scaler.transform(features)
        return self.classifier.predict(features)

    def _extract_features_batch(self, images):
        """複数画像から特徴量を抽出する"""
        return np.array([self._extract_features(img) for img in images])

    def get_feature_names(self):
        """
        特徴量名のリストを返す
        
        Returns:
            特徴量名のリスト
        """
        if self._feature_names is None:
            # 特徴量名がまだ生成されていない場合は、ダミー画像で生成
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            self._extract_features(dummy_image)
        # この時点でself._feature_namesはNoneではない
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
                            feature_names.append(f"{name}_{i}" if len(value) > 1 else name)
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
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # 輪郭が見つからない場合はゼロで埋める
            self._add_features(features, feature_names, {
                "circularity": 0,
                "area_ratio": 0,
                "aspect_ratio": 1,
                "extent": 0,
                "solidity": 0,
                "complexity": 0,
                "hu_1": 0,
                "hu_2": 0,
                "hu_3": 0,
                "hu_4": 0,
            })
        else:
            # 最大の輪郭を取得
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            # 円形度: 真円に近いほど1に近づく
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 面積比: 画像全体に対する輪郭の面積
            area_ratio = area / (gray.shape[0] * gray.shape[1])

            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0

            # 凸包との比較
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # 複雑さ
            complexity = perimeter / np.sqrt(area) if area > 0 else 0

            # Huモーメント（形状の特徴を表す不変量）
            moments = cv2.moments(largest)
            hu = cv2.HuMoments(moments).flatten()[:4]
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            self._add_features(features, feature_names, {
                "circularity": circularity,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "solidity": solidity,
                "complexity": complexity,
                "hu_1": hu[0],
                "hu_2": hu[1],
                "hu_3": hu[2],
                "hu_4": hu[3],
            })

        # ========================================
        # テクスチャ特徴量（LBP + Gabor）
        # ========================================
        # LBP (Local Binary Pattern): 局所的なテクスチャパターンを捉える
        # 各ピクセルの周囲8近傍と比較し、パターンをエンコード
        padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i in range(8):
            angle = i * np.pi / 4
            dy, dx = int(np.round(np.sin(angle))), int(np.round(np.cos(angle)))
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp += ((neighbor >= gray).astype(np.uint8) << i)
        # LBPヒストグラム（16ビンに正規化）
        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        hist_normalized = (hist / (hist.sum() + 1e-7)).tolist()
        self._add_features(features, feature_names, {
            f"lbp_{i}": hist_normalized[i] for i in range(16)
        })

        # Gaborフィルタ: 異なる方向・スケールのテクスチャを検出
        gabor_features = {}
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            theta_deg = int(np.degrees(theta))
            for sigma in [3.0, 5.0]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                gabor_features[f"gabor_{theta_deg}_{int(sigma)}_mean"] = filtered.mean()
                gabor_features[f"gabor_{theta_deg}_{int(sigma)}_std"] = filtered.std()
        self._add_features(features, feature_names, gabor_features)

        # ========================================
        # エッジ特徴量
        # ========================================
        # Cannyエッジ検出: エッジの密度を計算
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        # Sobelフィルタ: 勾配の強度と方向
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)

        # 勾配方向のヒストグラム（8方向）
        dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-7)

        # Laplacian: エッジの鮮明さ（分散が大きいほど鮮明）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000

        edge_features = {
            "edge_density": edge_density,
            "gradient_mean": gradient.mean(),
            "gradient_std": gradient.std(),
            "laplacian_var": laplacian_var,
        }
        # 方向ヒストグラムを追加
        for i in range(8):
            edge_features[f"dir_hist_{i}"] = dir_hist[i]
        self._add_features(features, feature_names, edge_features)

        # ========================================
        # 統計的特徴量
        # ========================================
        # 基本統計量
        mean, std = gray.mean(), gray.std()

        # エントロピー（情報量の尺度）
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # 歪度と尖度（分布の形状）
        centered = gray.astype(np.float64) - mean
        skewness = np.mean(centered ** 3) / (std ** 3 + 1e-7)
        kurtosis = np.mean(centered ** 4) / (std ** 4 + 1e-7) - 3

        # パーセンタイル
        percentiles = np.percentile(gray.ravel(), [10, 25, 50, 75, 90]) / 255

        self._add_features(features, feature_names, {
            "intensity_mean": mean / 255,
            "intensity_std": std / 255,
            "intensity_range": (gray.max() - gray.min()) / 255,
            "entropy": entropy / 8,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "pct_10": percentiles[0],
            "pct_25": percentiles[1],
            "pct_50": percentiles[2],
            "pct_75": percentiles[3],
            "pct_90": percentiles[4],
        })

        # 特徴量名をキャッシュ
        if feature_names is not None:
            self._feature_names = feature_names

        return np.array(features)

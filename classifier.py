import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class SnowCrystalClassifier:
    """
    OpenCVを使用した画像処理特徴量に基づく雪晶分類器

    画像から形状・テクスチャ・エッジ・統計的特徴量を抽出し、
    RandomForestで霰(graupel)と雪片(snowflake)を分類する
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        use_class_weights: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: 決定木の数
            max_depth: 決定木の最大深さ
            use_class_weights: クラス重みを使用するかどうか
            random_state: 乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_class_weights = use_class_weights
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = self._create_classifier()
        self.classes_ = np.array([0, 1])  # 0: graupel, 1: snowflake

    def _create_classifier(self) -> RandomForestClassifier:
        """分類器を作成する"""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced" if self.use_class_weights else None,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SnowCrystalClassifier":
        """
        モデルを訓練する

        Args:
            X_train: 訓練データ (N, H, W, C)
            y_train: 訓練ラベル (N,)

        Returns:
            self
        """
        features = self._extract_features_batch(X_train)
        features = self.scaler.fit_transform(features)
        self.classifier.fit(features, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を行う

        Args:
            X: 入力データ (N, H, W, C)

        Returns:
            予測ラベル (N,)
        """
        features = self._extract_features_batch(X)
        features = self.scaler.transform(features)
        return self.classifier.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う

        Args:
            X: 入力データ (N, H, W, C)

        Returns:
            各クラスの確率 (N, 2)
        """
        features = self._extract_features_batch(X)
        features = self.scaler.transform(features)
        return np.asarray(self.classifier.predict_proba(features))

    def _extract_features_batch(self, images: np.ndarray) -> np.ndarray:
        """バッチ画像から特徴量を抽出する"""
        return np.array([self._extract_features(img) for img in images])

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """単一画像から特徴量を抽出する"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()

        features = []
        features.extend(self._extract_shape_features(gray))
        features.extend(self._extract_texture_features(gray))
        features.extend(self._extract_edge_features(gray))
        features.extend(self._extract_statistical_features(gray))
        return np.array(features)

    def _extract_shape_features(self, gray: np.ndarray) -> list:
        """形状特徴量を抽出する"""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # 基本形状特徴量
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        area_ratio = area / (gray.shape[0] * gray.shape[1])

        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        complexity = perimeter / np.sqrt(area) if area > 0 else 0

        # Huモーメント（最初の4つ）
        moments = cv2.moments(largest)
        hu = cv2.HuMoments(moments).flatten()[:4]
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        return [circularity, area_ratio, aspect_ratio, extent, solidity, complexity] + hu.tolist()

    def _extract_texture_features(self, gray: np.ndarray) -> list:
        """テクスチャ特徴量（LBP + Gabor）を抽出する"""
        features = []

        # LBP
        padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i in range(8):
            angle = i * np.pi / 4
            dy, dx = int(np.round(np.sin(angle))), int(np.round(np.cos(angle)))
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp += ((neighbor >= gray).astype(np.uint8) << i)
        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        features.extend((hist / (hist.sum() + 1e-7)).tolist())

        # Gabor
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for sigma in [3.0, 5.0]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                features.extend([filtered.mean(), filtered.std()])

        return features

    def _extract_edge_features(self, gray: np.ndarray) -> list:
        """エッジ特徴量を抽出する"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)

        dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-7)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000

        return [edge_density, gradient.mean(), gradient.std(), laplacian_var] + dir_hist.tolist()

    def _extract_statistical_features(self, gray: np.ndarray) -> list:
        """統計的特徴量を抽出する"""
        mean, std = gray.mean(), gray.std()
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        centered = gray.astype(np.float64) - mean
        skewness = np.mean(centered ** 3) / (std ** 3 + 1e-7)
        kurtosis = np.mean(centered ** 4) / (std ** 4 + 1e-7) - 3

        percentiles = np.percentile(gray.ravel(), [10, 25, 50, 75, 90]) / 255

        return [
            mean / 255, std / 255, (gray.max() - gray.min()) / 255,
            entropy / 8, skewness, kurtosis
        ] + percentiles.tolist()

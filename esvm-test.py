import cv2
from skimage.feature import hog
import numpy as np
from sklearn.svm import SVC
import os

def extract_hog_features(image_path, max_length=250000):
    image = cv2.imread(image_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        # 填充或截断操作
        if len(features) < max_length:
            features = np.pad(features, (0, max_length - len(features)), 'constant')
        elif len(features) > max_length:
            features = features[:max_length]
        return features
    else:
        print(f"无法加载图像：{image_path}")
        return None

# 定义正样本路径和负样本路径
positive_sample_folder = 'A'
negative_sample_folder = 'C'

# 提取负样本特征
X_train_neg = []
for filename in os.listdir(negative_sample_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        sample_path = os.path.join(negative_sample_folder, filename)
        features = extract_hog_features(sample_path)
        if features is not None:
            X_train_neg.append(features)

# 填充或截断正样本特征并训练Exemplar SVM
for filename in os.listdir(positive_sample_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        sample_path = os.path.join(positive_sample_folder, filename)
        features = extract_hog_features(sample_path)
        if features is not None:
            # 构建训练集
            X_train_pos = np.array([features] * len(X_train_neg))
            X_train = np.vstack((X_train_pos, np.array(X_train_neg)))

            y_train = np.array([1] * len(X_train_pos) + [-1] * len(X_train_neg))

            # 实例化ExemplarSVM模型
            model = SVC(kernel='linear')

            # 训练ExemplarSVM模型
            model.fit(X_train, y_train)

            # 打印分类器的权重值
            print(f"Positive sample {filename} 对应的分类器权重值: {model.coef_[0]}")


import os
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import io, transform
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = io.imread(image_path)
    
    # Preprocess the image (e.g., resizing, gray scaling)
    image = transform.resize(image, (256, 128))  # set size
    image = (image * 255).astype(np.uint8)
    
    # Extract HOG features
    features, _ = hog(image, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True, multichannel=True)
    return features

def train_exemplar_svm(training_data, labels):
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(training_data)
    
    # Train SVM
    model = svm.LinearSVC()  # You can adjust parameters as needed
    model.fit(scaled_features, labels)
    
    # Retrieve the trained weight vector W
    W = model.coef_
    return W, scaler

def load_images_from_folder(folder_path):
    images = []
    labels = []
    # List all files in the directory and read them as images
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            # Preprocess and extract features from the image
            img_features = preprocess_image(img_path)
            images.append(img_features)
            labels.append(1)  # Assuming all images are positive examples
    return images, labels

# Define your image folder path
image_folder_path = 'office_caltech_10/amazon/back_pack'

# Load and preprocess all images in the folder
features, labels = load_images_from_folder(image_folder_path)

# Train the SVM and retrieve initial W and the scaler
W_initial, feature_scaler = train_exemplar_svm(features, labels)

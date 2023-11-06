from pathlib import Path
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import Bunch

def load_image_data(image_path, image_size=(64, 64)):
    image_directory = Path(image_path)
    sub_dirs = [sub_dir for sub_dir in image_directory.iterdir() if sub_dir.is_dir()]
    category_names = [folder.name for folder in sub_dirs]

    description = "An image classification dataset"
    dataset_images = []
    dataset_flat_data = []
    dataset_labels = []
    for label, sub_dir in enumerate(sub_dirs):
        for file in sub_dir.iterdir():
            image = imread(file)
            image_resized = resize(image, image_size, anti_aliasing=True, mode='reflect')
            dataset_flat_data.append(image_resized.flatten())
            dataset_images.append(image_resized)
            dataset_labels.append(label)
    dataset_flat_data = np.array(dataset_flat_data)
    dataset_labels = np.array(dataset_labels)
    dataset_images = np.array(dataset_images)

    return Bunch(data=dataset_flat_data,
                 target=dataset_labels,
                 target_names=category_names,
                 images=dataset_images,
                 DESCR=description)

# Load dataset
dataset = load_image_data("office_caltech_10/amazon")

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=109)

# Define parameter grid for GridSearch
parameter_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Initialize the SVM classifier and GridSearch
classifier = svm.SVC()
grid_search = GridSearchCV(classifier, parameter_grid)
grid_search.fit(X_train, y_train)

# Predict on the test set
predictions = grid_search.predict(X_test)

# Print the classification report
print(f"Classification report for classifier {grid_search}:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")

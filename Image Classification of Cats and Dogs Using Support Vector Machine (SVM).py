import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm

# Paths
train_data_dir = '/Users/virinchisai/Downloads/PROJECTS/Prodigy Infotech/PRODIGY_ML_03/dogs-vs-cats/train'
test_data_dir = '/Users/virinchisai/Downloads/PROJECTS/Prodigy Infotech/PRODIGY_ML_03/dogs-vs-cats/test1'

# Image dimensions
img_width, img_height = 128, 128  # Reduced resolution for faster processing

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Function to load and preprocess the dataset
def load_and_preprocess_image(filepath):
    img = load_img(filepath, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features using VGG16
def extract_feature(file, directory):
    if file.endswith(".jpg"):
        filepath = os.path.join(directory, file)
        img_array = load_and_preprocess_image(filepath)
        feature = model.predict(img_array)
        label = 1 if "dog" in file else 0  # 1 for dog, 0 for cat
        return feature.flatten(), label
    return None

# Function to extract features from directory using parallel processing
def extract_features_parallel(directory, n_jobs=4):
    files = os.listdir(directory)
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(extract_feature)(file, directory) for file in tqdm(files))
    results = [result for result in results if result is not None]  # Filter out None results
    features, labels = zip(*results)
    return np.array(features), np.array(labels)

# Extract features from the training dataset
print("Extracting features from training dataset...")
X_train, y_train = extract_features_parallel(train_data_dir, n_jobs=4)
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

# Train an SVM classifier
print("Training SVM model...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
print("SVM model trained successfully.")

# Extract features from the test dataset
print("Extracting features from test dataset...")
X_test, test_filenames = extract_features_parallel(test_data_dir, n_jobs=4)
print(f"Test data shape: {X_test.shape}")

# Make predictions on the test dataset
print("Making predictions on the test dataset...")
y_pred = svm.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "id": [os.path.splitext(file)[0] for file in test_filenames],
    "label": y_pred
})

submission_file_path = "submission.csv"
submission.to_csv(submission_file_path, index=False)
print(f"Predictions saved to {submission_file_path}")

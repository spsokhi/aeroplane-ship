from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data = []
labels = []
classes = 2
cur_path = os.getcwd()  # To get current directory

classs = {0: "Aeroplane", 1: "Ship"}

# Retrieving the images and their labels
print("Obtaining Images & its Labels..............")
for i in range(0, classes ):  # Start from 1 instead of 0
    path = os.path.join(cur_path, 'dataset/train/', str(i))

    if not os.path.exists(path):
        print(f"Directory not found: {path}")
    else:
        images = os.listdir(path)

        if not images:
            print(f"No images found in {path}")
        else:
            for a in images:
                try:
                    image = Image.open(os.path.join(path, a))
                    image = image.resize((30, 30))
                    image = np.array(image)
                    data.append(image)
                    labels.append(i)
                except Exception as e:
                    print(f"Error loading image {a}: {e}")

print("Dataset Loaded")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

def classify(img_file):
    model = load_model('my_model.h5')
    print("Loaded model from disk")
    path2 = img_file
    print(path2)
    test_image = Image.open(path2)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)
    predict_x = model.predict(test_image)
    result = np.argmax(predict_x, axis=1)
    sign = classs[int(result)]
    print(sign)

# Specify the path to your test images
test_path = 'dataset/test/'
files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith('.jpg')]

for f in files:
    classify(f)
    print('\n')


# 🚦 Traffic Jam Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify traffic images into two categories: `JAM` and `NOT_JAM`. The model is trained using TensorFlow and Keras in Google Colab. The dataset is stored in Google Drive and consists of images divided into training and testing sets.

---

## 📁 Dataset Structure

The dataset should be organized as follows:

```
Traffic_Dataset/
│
├── train/             # 80% of data
│   ├── JAM/
│   └── NOT_JAM/
│
└── test/              # 20% of data
    ├── JAM/
    └── NOT_JAM/
```

---

## ✅ Step 1: Data Preparation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths (after mounting)
base_dir = "/content/drive/MyDrive/Traffic_Dataset"
train_dir = f"{base_dir}/train"
test_dir = f"{base_dir}/test"

# Image preprocessing
img_size = (128, 128)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
```

---

## ✅ Step 2: CNN Model Creation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## ✅ Step 3: Model Training

```python
model.fit(train_generator, validation_data=test_generator, epochs=30)
```

---

## ✅ Step 4: Save the Model

```python
model.save("/content/drive/MyDrive/traffic_jam_classifier.keras")
print("✅ Model saved as traffic_jam_classifier.keras")
```

---

## ✅ Step 5: Test on a Single Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load the model
model = load_model("/content/drive/MyDrive/traffic_jam_classifier.keras")

# ✅ Path to the test image (update the path if needed)
img_path = "/content/drive/MyDrive/Traffic_Dataset/test/NOT_JAM/not_jam140.JPG"

# ✅ Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ✅ Make prediction
prediction = model.predict(img_array)[0][0]
predicted_label = "NOT_JAM" if prediction > 0.5 else "JAM"

# ✅ Show the image with predicted label and confidence
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}\nConfidence: {prediction:.2f}")
plt.axis('off')
plt.show()
```

---

## ✅ Step 6: Predict Random Images from Each Class

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Load the trained model
model = load_model("/content/drive/MyDrive/traffic_jam_classifier.keras")

# ✅ Paths to test directories
test_base = "/content/drive/MyDrive/Traffic_Dataset/test"
categories = ["JAM", "NOT_JAM"]

# ✅ Function to load, preprocess, and predict a single image
def predict_random_image_from_category(category):
    folder_path = os.path.join(test_base, category)
    img_name = random.choice(os.listdir(folder_path))
    img_path = os.path.join(folder_path, img_name)

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    predicted_label = "NOT_JAM" if prediction > 0.5 else "JAM"

    return img, img_path, predicted_label, prediction

# ✅ Plot predictions for one image from each class
plt.figure(figsize=(12, 6))

for i, category in enumerate(categories):
    img, path, pred_label, conf = predict_random_image_from_category(category)
    plt.subplot(1, 2, i+1)
    plt.imshow(img)
    plt.title(f"Actual: {category}\nPredicted: {pred_label}\nConfidence: {conf:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 📊 Results

- The CNN model classifies traffic images with a binary output: **JAM** or **NOT_JAM**.
- It is trained and tested on a dataset split 80/20.
- Final model is saved as `.keras` and used for future predictions.

---

## 🚀 Future Improvements

- Add real-time video or webcam support.
- Apply data augmentation to reduce overfitting.
- Try using transfer learning (e.g., MobileNet, ResNet).
- Deploy the model using a web interface (Flask or Streamlit).

# 🧠 Train and Test Traffic Jam Classifier in Anaconda Jupyter Notebook

This guide describes how to train and evaluate a Convolutional Neural Network (CNN) model locally using **Anaconda Jupyter Notebook** for classifying traffic images into **JAM** and **NOT_JAM** categories.

---

## 🛠️ Setup

Make sure you have the following packages installed in your Anaconda environment:

- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook
- PIL (Pillow)

To install them (if not installed), use:

```bash
pip install tensorflow numpy matplotlib pillow
```

---

## 📁 Directory Structure

```
Traffic_CNN_Training/
│
├── Traffic_Dataset/
│   ├── train/
│   │   ├── JAM/
│   │   └── NOT_JAM/
│   └── test/
│       ├── JAM/
│       └── NOT_JAM/
│
└── traffic_jam_classifier.keras  # Will be saved after training
```

---

## 📒 Jupyter Notebook Steps

### ✅ Step 1: Import Required Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
```

### ✅ Step 2: Load Dataset

```python
# Paths
train_dir = "Traffic_Dataset/train"
test_dir = "Traffic_Dataset/test"

# Image size and batch size
img_size = (128, 128)
batch_size = 32

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset using generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
```

### ✅ Step 3: Build CNN Model

```python
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### ✅ Step 4: Compile the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### ✅ Step 5: Train the Model

```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
```

### ✅ Step 6: Save the Model

```python
model.save("traffic_jam_classifier.keras")
print("✅ Model saved as 'traffic_jam_classifier.keras'")
```

### ✅ Step 7: Plot Accuracy & Loss

```python
# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
```

### ✅ Step 8: Predict on Random Test Images

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1) Paths – adjust to your local setup
base_dir       = r"C:\Users\jmtam\Desktop\Traffic_CNN_Training\Traffic_Dataset\test"
jam_folder     = os.path.join(base_dir, "JAM")
not_jam_folder = os.path.join(base_dir, "NOT_JAM")

# 2) Pick 4 random images from each
jam_images     = random.sample([f for f in os.listdir(jam_folder)     if f.lower().endswith(('.jpg','.jpeg','.png'))], 4)
not_jam_images = random.sample([f for f in os.listdir(not_jam_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))], 4)

# 3) Build list of (path, actual_label)
image_paths = ([(os.path.join(jam_folder, img),     'JAM')     for img in jam_images] +
               [(os.path.join(not_jam_folder, img), 'NOT_JAM') for img in not_jam_images])

# 4) Load your trained model
model = load_model(r"C:\Users\jmtam\Desktop\Traffic_CNN_Training\traffic_jam_classifier.keras")

# 5) Warm‑up to avoid retracing (optional)
_ = model.predict(np.zeros((1,128,128,3)))

# 6) Plot and predict
plt.figure(figsize=(16, 8))
for i, (img_path, actual) in enumerate(image_paths):
    # preprocess for model
    img128 = image.load_img(img_path, target_size=(128,128))
    arr    = image.img_to_array(img128)[np.newaxis,...] / 255.0
    pred   = model.predict(arr, verbose=0)[0][0]
    
    # determine labels & confidence
    if pred > 0.5:
        predicted, conf = "NOT_JAM", pred
    else:
        predicted, conf = "JAM", 1 - pred

    # display original for clarity
    orig = image.load_img(img_path)
    plt.subplot(2, 4, i+1)
    plt.imshow(orig, interpolation='nearest')
    plt.title(f"True: {actual}\nPred: {predicted}\nConf: {conf:.2%}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### ✅ Step 9: Predict a Specific Image

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model(r"C:\Users\jmtam\Desktop\Traffic_CNN_Training\traffic_jam_classifier.keras")

# Specify the image path
img_path = r"C:\Users\jmtam\Desktop\Traffic_CNN_Training\Traffic_Dataset\test\NOT_JAM\not_jam153.JPG"

# Verify the file exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# Load & preprocess for prediction
img128 = image.load_img(img_path, target_size=(128,128))
arr = image.img_to_array(img128)[np.newaxis, ...] / 255.0

# Predict
pred = model.predict(arr, verbose=0)[0][0]
if pred > 0.5:
    label, confidence = "NOT_JAM", pred
else:
    label, confidence = "JAM", 1 - pred

# Print results
print(f"Prediction: {label}")
print(f"Confidence: {confidence * 100:.2f}%")

# Display the original image with title
orig = image.load_img(img_path)
plt.figure(figsize=(5,5))
plt.imshow(orig)
plt.title(f"{label} ({confidence*100:.2f}%)")
plt.axis('off')
plt.show()
```

---

✅ That's it! You have now successfully trained and evaluated your traffic jam detection model in a local Anaconda Jupyter Notebook.

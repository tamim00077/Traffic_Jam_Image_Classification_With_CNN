# Traffic Jam Classification using CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify traffic images into two categories: "JAM" and "NOT_JAM". The model helps to automatically identify traffic congestion from images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Process](#training-process)
  - [Model Evaluation](#model-evaluation)
- [Results and Visualization](#results-and-visualization)
- [Using the Model](#using-the-model)
- [Future Improvements](#future-improvements)

## Project Overview

Traffic congestion detection is crucial for modern traffic management systems. This project uses deep learning to automate the classification of traffic images, determining whether they show congested (JAM) or free-flowing (NOT_JAM) traffic conditions.

The implementation uses TensorFlow and Keras to build a CNN model, which is trained on a dataset of labeled traffic images.

## Dataset Description

The dataset consists of traffic images organized as follows:

```
Traffic_Dataset/
├── train/
│   ├── JAM/
│   └── NOT_JAM/
└── test/
    ├── JAM/
    └── NOT_JAM/
```

- **Training set (80%)**: Used to train the model
- **Testing set (20%)**: Used to evaluate model performance

Each image is categorized into one of two classes:
- **JAM**: Images showing congested traffic
- **NOT_JAM**: Images showing free-flowing traffic

## Environment Setup

This project was implemented in Google Colab with the following dependencies:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
```

## Project Structure

```
project/
├── Traffic_Dataset/         # Dataset directory
│   ├── train/
│   │   ├── JAM/             # Training images of traffic jams
│   │   └── NOT_JAM/         # Training images of normal traffic
│   └── test/
│       ├── JAM/             # Testing images of traffic jams
│       └── NOT_JAM/         # Testing images of normal traffic
├── traffic_jam_classifier.keras  # Saved model
└── notebooks/
    └── traffic_jam_classification.ipynb  # Main implementation notebook
```

## Implementation Details

### Data Preprocessing

The images are processed using Keras' `ImageDataGenerator` to:
1. Rescale pixel values from [0-255] to [0-1]
2. Resize all images to 128×128 pixels
3. Load images in batches of 32

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

Note: For more robust models, you could add data augmentation techniques like rotation, width/height shifts, zooming, etc.

### Model Architecture

The CNN model consists of:
- 2 convolutional layers with max-pooling
- Flattening layer to convert 2D feature maps to 1D features
- Dense hidden layer with dropout for regularization
- Output layer with sigmoid activation for binary classification

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

The architecture breakdown:
1. First Conv2D layer: 32 filters, 3×3 kernel, ReLU activation
2. First MaxPooling: 2×2 pooling to reduce spatial dimensions
3. Second Conv2D layer: 64 filters, 3×3 kernel, ReLU activation
4. Second MaxPooling: 2×2 pooling
5. Flatten: Convert 3D feature maps to 1D feature vector
6. Dense layer: 128 neurons with ReLU activation
7. Dropout: 50% dropout rate to prevent overfitting
8. Output: Single neuron with sigmoid activation (0 = JAM, 1 = NOT_JAM)

### Training Process

The model was trained for 30 epochs using the Adam optimizer and binary cross-entropy loss function:

```python
model.fit(train_generator, validation_data=test_generator, epochs=30)
```

After training, the model was saved to disk:

```python
model.save("/content/drive/MyDrive/traffic_jam_classifier.keras")
print("✅ Model saved as traffic_jam_classifier.keras")
```

### Model Evaluation

The model's performance was evaluated on the test dataset, and predictions were visualized to qualitatively assess performance.

## Results and Visualization

Example predictions on random test images:

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

## Using the Model

To use the trained model for prediction on new images:

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

## Future Improvements

Several enhancements could be made to improve the model:

1. **Data Augmentation**: Implement techniques like rotation, zoom, flip, etc., to increase dataset diversity
2. **Model Architecture**: Experiment with deeper architectures or pre-trained models (e.g., VGG16, ResNet)
3. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
4. **Class Imbalance**: Address any imbalance in the dataset
5. **Explainability**: Add visualization techniques (e.g., Grad-CAM) to highlight areas the model focuses on

## License

[J.M. Tamimur Rahman]

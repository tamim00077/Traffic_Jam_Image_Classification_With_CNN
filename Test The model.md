
# Traffic Jam Classifier - Model Evaluation

This repository contains code to evaluate a trained CNN model (`traffic_jam_classifier.keras`) for traffic jam classification using test images. It includes metrics such as confusion matrix, classification report, ROC curve, precision-recall curve, and convergence plots.

---

## 📁 Directory Structure

```
Traffic_Dataset/
└── test/
    ├── Jam/
    └── No_Jam/
```

---

## 🧾 Dependencies

Make sure the following libraries are installed:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow
```

---

## 📌 Evaluation Code

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the trained model

model = load_model("traffic_jam_classifier.keras")

# Parameters

img_size = (128, 128)
batch_size = 32
test_dir = "Traffic_Dataset/test"

# Test data generator

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=img_size,
batch_size=batch_size,
class_mode='binary',
shuffle=False  # Important for label consistency
)

# Get predictions

test_generator.reset()
Y_true = test_generator.classes
Y_pred_prob = model.predict(test_generator)
Y_pred = (Y_pred_prob > 0.5).astype(int).reshape(-1)

# --- Confusion Matrix ---

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Jam', 'Jam'], yticklabels=['No Jam', 'Jam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Classification Report ---

report = classification_report(Y_true, Y_pred, target_names=['No Jam', 'Jam'])
print("Classification Report:
", report)
```

---

## 📈 ROC Curve and Precision-Recall Curve

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize the true labels

Y_true_bin = label_binarize(Y_true, classes=[0, 1])

# Class 0 = No Jam, Class 1 = Jam

Y_prob_no_jam = 1 - Y_pred_prob
Y_prob_jam = Y_pred_prob

# ----------- ROC CURVES FOR BOTH CLASSES AND OVERALL MODEL -----------

fpr_0, tpr_0, _ = roc_curve(1 - Y_true_bin[:, 0], Y_prob_no_jam)
roc_auc_0 = auc(fpr_0, tpr_0)

fpr_1, tpr_1, _ = roc_curve(Y_true_bin[:, 0], Y_prob_jam)
roc_auc_1 = auc(fpr_1, tpr_1)

# Overall ROC

fpr, tpr, _ = roc_curve(Y_true_bin[:, 0], Y_pred_prob)
roc_auc = auc(fpr, tpr)

# ----------- PRECISION-RECALL CURVES FOR BOTH CLASSES AND OVERALL MODEL -----------

precision_0, recall_0, _ = precision_recall_curve(1 - Y_true_bin[:, 0], Y_prob_no_jam)
pr_auc_0 = auc(recall_0, precision_0)

precision_1, recall_1, _ = precision_recall_curve(Y_true_bin[:, 0], Y_prob_jam)
pr_auc_1 = auc(recall_1, precision_1)

# Overall Precision-Recall

precision, recall, _ = precision_recall_curve(Y_true_bin[:, 0], Y_pred_prob)
pr_auc = auc(recall, precision)

# ----- ROC Curve Plot -----

plt.figure(figsize=(12, 6))

# Plot ROC for No Jam

plt.subplot(1, 2, 1)
plt.plot(fpr_0, tpr_0, label=f'No Jam (AUC = {roc_auc_0:.2f})', color='blue')
plt.plot(fpr_1, tpr_1, label=f'Jam (AUC = {roc_auc_1:.2f})', color='green')
plt.plot(fpr, tpr, label=f'Overall Model (AUC = {roc_auc:.2f})', color='red', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

# ----- Precision-Recall Curve Plot -----

plt.subplot(1, 2, 2)
plt.plot(recall_0, precision_0, label=f'No Jam (AUC = {pr_auc_0:.2f})', color='blue')
plt.plot(recall_1, precision_1, label=f'Jam (AUC = {pr_auc_1:.2f})', color='green')
plt.plot(recall, precision, label=f'Overall Model (AUC = {pr_auc:.2f})', color='red', linestyle='--')
plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 📉 Convergence Curve (Epoch vs Accuracy & Loss)

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

---

## 📌 Notes

- Ensure the model file `traffic_jam_classifier.keras` is in the same directory as the script.
- Make sure the test dataset is structured correctly.
- The convergence plot requires a `history` object from training.

---

## 🧠 Author

This README is based strictly on the original code provided with no changes.


# 🖥️ Running the Traffic Jam Classifier Locally

This guide walks you through the steps to run the trained CNN traffic jam classification model locally on your computer using Flask.

---

## 📁 Project Structure

Make sure your files and folders are arranged as follows:

```
project_folder/
│
├── traffic_jam_classifier.keras
├── app.py
├── static/
│   └── uploads/          # Uploaded images will be saved here
│
├── templates/
    ├── index.html        # Form to upload image
    └── result.html       # Shows prediction result
```

---

## ⚙️ Required Libraries

Ensure you have the following Python libraries installed:

```bash
pip install tensorflow flask pillow numpy
```

---

## 🚀 Running the App

1. Open a terminal or command prompt.
2. Navigate to the folder where `app.py` is located.
3. Run the following command:

```bash
python app.py
```

4. Open your browser and go to `http://127.0.0.1:5000`

You should now be able to upload a traffic image and get a prediction (Jam or Not Jam) along with confidence.

---

## 📝 Provided Scripts

### 🔹 app.py

```python
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('traffic_jam_classifier.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0][0]
        label = 'Not Jam' if prediction > 0.5 else 'Jam'
        confidence = prediction if prediction > 0.5 else 1 - prediction

        image_path = f"/uploads/{filename}"
        return render_template('result.html', label=label, confidence=f"{confidence:.2f}", image_path=image_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
```

### 🔹 index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Traffic Jam Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <h2 class="text-center mb-4">🚦 Traffic Jam Detection</h2>
        <div class="card p-4 shadow-sm">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="mb-3">
                    <label for="image" class="form-label">Upload an Image:</label>
                    <input class="form-control" type="file" name="image" id="image" required>
                </div>
                <button type="submit" class="btn btn-primary w-100">Predict</button>
            </form>
        </div>
    </div>
</body>
</html>
```

### 🔹 result.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <h2 class="text-center mb-4">🧠 Prediction Result</h2>
        <div class="card shadow-sm mx-auto" style="width: 22rem;">
            <img src="{{ image_path }}" class="card-img-top" alt="Uploaded Image">
            <div class="card-body text-center">
                <h5 class="card-title">Class: {{ label }}</h5>
                <p class="card-text">Confidence: {{ confidence }}</p>
                <a href="/" class="btn btn-secondary">Try Another</a>
            </div>
        </div>
    </div>
</body>
</html>
```

---

## ✅ You're all set!
You can now use the trained CNN model in a user-friendly local web interface to classify traffic images.

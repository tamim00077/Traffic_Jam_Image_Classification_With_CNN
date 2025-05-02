
# 🖼️ Image Preprocessing for Traffic Jam Classification

This README documents the preprocessing steps done before training the CNN model to classify traffic images into `JAM` and `NOT_JAM`. The steps include renaming files, resizing/compressing images, and splitting them into training and testing sets.

---

## 🔠 Step 1: Renaming Image Files

### ✅ Rename all images in a folder

```python
import os

folder_path = "."
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

for index, filename in enumerate(files, start=1):
    extension = os.path.splitext(filename)[1]
    new_name = f"jam{index}{extension}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    print(f"Renamed {filename} to {new_name}")
```

- Save this script as `rename.py`
- Put it in the folder containing the images.
- Open **CMD** in that folder and run:

```bash
python rename.py
```

---

### ✅ Rename starting from a certain number (e.g., from 101)

```python
import os

folder_path = "."
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

for index, filename in enumerate(files, start=101):
    extension = os.path.splitext(filename)[1]
    new_name = f"jam{index}{extension}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    print(f"Renamed {filename} to {new_name}")
```

---

## 📉 Step 2: Reduce/Compress Image Size

```python
import os
from PIL import Image, ImageOps

# Set the input and output folder paths
input_folder = r"D:\Data mining\Dataset"      # Folder containing original images
output_folder = r"D:\Data mining\Compressed"  # Folder to save compressed images

# Set the desired image quality (0-100), where lower means more compression
quality = 20 # Recommended: 70 for good balance between quality and size

# Create the output root directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"🔍 Scanning images in: {input_folder}")
print(f"💾 Compressed images will be saved to: {output_folder}\n")

# Walk through all subdirectories and files in the input folder
for dirpath, _, filenames in os.walk(input_folder):
    # Compute the relative path from the input folder
    relative_path = os.path.relpath(dirpath, input_folder)
    # Create the corresponding output subdirectory
    output_subfolder = os.path.join(output_folder, relative_path)
    os.makedirs(output_subfolder, exist_ok=True)

    # Process each file in the current directory
    for filename in filenames:
        # Check if the file is an image based on its extension
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_file = os.path.join(dirpath, filename)          # Full path to the input image
            output_file = os.path.join(output_subfolder, filename)  # Full path to save the compressed image

            try:
                with Image.open(input_file) as img:
                    # Adjust image orientation based on EXIF data
                    img = ImageOps.exif_transpose(img)
                    # Convert image to RGB mode to ensure compatibility with JPEG format
                    img = img.convert("RGB")
                    # Save the image with the specified quality and optimization
                    img.save(output_file, format='JPEG', optimize=True, quality=quality)
                print(f"✅ Compressed: {input_file} → {output_file}")
            except Exception as e:
                print(f"⚠️ Error processing {input_file}: {e}")

print("\n🎉 All images have been successfully compressed.")
```

### 🛠️ Usage Instructions:
1. Put `reducescript.py` beside the dataset folder (which contains `jam` and `not jam` folders).
2. Open **CMD** in that directory.
3. Run the script:

```bash
python reducescript.py
```

4. Press Enter.

---

## ✂️ Step 3: Split Dataset (80% Train, 20% Test)

```python
import os
import shutil
import random

# Automatically find the "Traffic dataset" folder in the current directory
BASE_DIR = os.getcwd()
SOURCE_DIR = os.path.join(BASE_DIR, "Traffic dataset")
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
TEST_DIR   = os.path.join(BASE_DIR, "test")
SPLIT_RATIO = 0.8

# Function to split into train/test
def split_dataset(source_dir, train_dir, test_dir, split_ratio=SPLIT_RATIO):
    # create output dirs if missing
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    for label in os.listdir(source_dir):
        class_path = os.path.join(source_dir, label)
        if not os.path.isdir(class_path):
            continue

        # shuffle files
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(files)

        # split point
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        # ensure class subfolders exist
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir,  label), exist_ok=True)

        # copy
        for fname in train_files:
            shutil.copy(os.path.join(class_path, fname),
                        os.path.join(train_dir, label, fname))
        for fname in test_files:
            shutil.copy(os.path.join(class_path, fname),
                        os.path.join(test_dir,  label, fname))

    print(f"✅ Done! Train samples in '{train_dir}', test samples in '{test_dir}'")

# Run split
split_dataset(SOURCE_DIR, TRAIN_DIR, TEST_DIR)
```

### 🛠️ Usage Instructions:
1. Put the `splitting_image.py` file beside the main folder named `Traffic dataset`.
2. Open **CMD** in that directory.
3. Run:

```bash
python splitting_image.py
```

---

## ✅ Output

- Your dataset will now be organized into `train/` and `test/` folders.
- Images will be renamed, compressed, and neatly sorted.

---

## 🧩 Note

These steps help ensure uniformity, reduce image loading time, and improve training efficiency.


# FInish.

# Additional Codes to handle HEIC:
# Image Compression Script (with HEIC Support)

This Python script compresses `.jpg`, `.jpeg`, `.png`, and `.heic` image files in a folder called `images` and saves the compressed `.jpg` versions in a `compressed` folder — all relative to the script's location.

---

## 📌 Features

- ✅ Supports **HEIC**, **JPG**, **JPEG**, and **PNG** formats.
- ✅ Converts all images to **optimized `.jpg`** files.
- ✅ Keeps everything **path-independent** — no configuration required.
- ✅ Works directly from the **command line**.
- ✅ Uses `pillow-heif` to handle HEIC files.

---

## 📂 Folder Structure

```
your-folder/
├── compress_images.py
├── images/
│   ├── photo1.jpg
│   ├── photo2.heic
│   └── ...
├── compressed/
```

---

## ⚙️ Requirements

Make sure Python and pip are installed, then run:

```bash
pip install pillow pillow-heif
```

---

## 🚀 How to Use

1. Place your images in the `images/` folder (create it if it doesn't exist).

2. Run the script using Python:
   ```bash
   python compress_images.py
   ```

3. Compressed `.jpg` files will appear in the `compressed/` folder.

---

## 🧠 The Script

```python
import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Enable HEIC format support
register_heif_opener()

# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set relative input and output folders
input_folder = os.path.join(base_dir, "images")
output_folder = os.path.join(base_dir, "compressed")

# Set compression quality (lower = smaller size, 70 is a good default)
quality = 70

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"🔍 Scanning: {input_folder}")
print(f"💾 Saving compressed images to: {output_folder}\n")

# Walk through images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.save(output_path, format='JPEG', optimize=True, quality=quality)
            print(f"✅ Compressed: {filename} → {output_filename}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

print("\n🎉 Done! All images have been compressed.")
```

---





# HEIC to JPG Converter (No Quality Loss)

This script converts `.heic` images to `.jpg` format while preserving maximum quality. It uses `pillow-heif` for HEIC support and `Pillow` for image processing.

---

## 📌 Features

- ✅ Converts `.heic` files to `.jpg` with **no visible quality loss**
- ✅ Automatically scans the `heic_images/` folder
- ✅ Saves `.jpg` files to a `jpg_images/` folder
- ✅ No hardcoded paths — portable and easy to run
- ✅ Command-line friendly

---

## 📂 Folder Structure

```
your-folder/
├── convert_heic_to_jpg.py
├── heic_images/
│   ├── photo1.heic
│   └── ...
├── jpg_images/
```

- Put your `.heic` files in the `heic_images/` folder
- Converted `.jpg` files will appear in the `jpg_images/` folder

---

## ⚙️ Requirements

Install the required libraries using pip:

```bash
pip install pillow pillow-heif
```

---

## 🚀 How to Use

1. Place `.heic` images in the `heic_images/` folder
2. Run the script:

```bash
python convert_heic_to_jpg.py
```

3. Check the `jpg_images/` folder for the converted `.jpg` files

---

## 🧠 The Script

```python
import os
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIC support
register_heif_opener()

# Set the input and output folders
input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heic_images")
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jpg_images")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"🔍 Converting HEIC images in: {input_folder}")
print(f"💾 Converted JPGs will be saved in: {output_folder}\n")

# Process each HEIC file
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.heic'):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")  # Ensure it's compatible with JPEG
                img.save(output_path, format="JPEG", quality=100)  # No quality loss
            print(f"✅ Converted: {filename} → {output_filename}")
        except Exception as e:
            print(f"⚠️ Failed to convert {filename}: {e}")

print("\n🎉 Done! All HEIC images converted to high-quality JPG.")
```

---


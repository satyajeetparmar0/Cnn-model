# Cnn-model
Designing a CNN-based model for breast cancer classification using histopathological images (BreakHis dataset).
# ✅ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import math

# === CONFIG ===
BASE_DIR = '/content/drive/MyDrive/dataset_cancer_v1/classificacao_binaria'
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']
IMG_SIZE = (64, 64)
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
IS_BINARY = True

# === LOAD IMAGES FROM ALL MAGNIFICATIONS ===
def load_dataset(base_dir, magnifications):
    images = []
    labels = []
    skipped = 0
    for mag in magnifications:
        mag_path = os.path.join(base_dir, mag)
        if not os.path.exists(mag_path): continue
        for label_name in os.listdir(mag_path):
            label_folder = os.path.join(mag_path, label_name)
            if not os.path.isdir(label_folder): continue
            for fname in os.listdir(label_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(label_folder, fname)
                    try:
                        img = load_img(img_path, target_size=IMG_SIZE)
                        img = img_to_array(img) / 255.0
                        images.append(img)
                        labels.append(label_name)
                    except UnidentifiedImageError:
                        skipped += 1
    print(f"❗ Skipped {skipped} corrupt/unreadable images.")
    return np.array(images), np.array(labels)

# === PAGINATED DISPLAY ===
def get_all_image_paths(base_path, magnifications):
    image_paths = []
    for mag in magnifications:
        mag_path = os.path.join(base_path, mag)
        if not os.path.exists(mag_path): continue
        for class_name in os.listdir(mag_path):
            class_path = os.path.join(mag_path, class_name)
            if not os.path.isdir(class_path): continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((os.path.join(class_path, fname), f"{class_name} - {mag}"))
    return image_paths

def display_images_paginated(image_paths, images_per_page=100):
    total = len(image_paths)
    pages = math.ceil(total / images_per_page)
    cols = 10
    rows = images_per_page // cols

    for p in range(pages):
        start = p * images_per_page
        end = min(start + images_per_page, total)
        print(f"📄 Showing images {start+1} to {end} of {total}")

        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
        fig.suptitle(f"🖼️ Dataset Images Page {p+1}/{pages}", fontsize=16)

        for i in range(images_per_page):
            r, c = divmod(i, cols)
            ax = axes[r][c] if rows > 1 else axes[c]
            idx = start + i
            if idx < total:
                img_path, label = image_paths[idx]
                try:
                    img = load_img(img_path, target_size=IMG_SIZE)
                    ax.imshow(img)
                    ax.set_title(label, fontsize=8)
                    ax.axis('off')
                except:
                    ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

# === LOAD DATA ===
print("🔄 Loading dataset...")
X, y = load_dataset(BASE_DIR, MAGNIFICATIONS)
print(f"✅ Loaded {len(X)} images.")

# === DISPLAY IMAGES ===
print("🖼️ Displaying all images by page...")
image_paths = get_all_image_paths(BASE_DIR, MAGNIFICATIONS)
display_images_paginated(image_paths, images_per_page=100)

# === LABEL ENCODING ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
loss_fn = 'binary_crossentropy'
final_activation = 'sigmoid'
output_units = 1

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === CNN MODEL ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(output_units, activation=final_activation)
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=['accuracy'])
model.summary()

# === TRAIN MODEL ===
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# === EVALUATE ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"📊 Test Accuracy: {test_acc:.2f}")

# === PLOT METRICS ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.legend()
plt.show()


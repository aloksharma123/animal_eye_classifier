import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =====================================
# Load Trained Model
# =====================================
MODEL_PATH = "model/animal_eye_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# =====================================
# Class Names
# (Must be in the same order as training)
# =====================================
class_names = [
    "Cat",
    "Dog",
    "Wild"
]

# =====================================
# Settings
# =====================================
TEST_FOLDER = "test_images"

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# Confidence threshold (80%)
CONFIDENCE_THRESHOLD = 0.80

print("=" * 60)
print("Animal Eye Classifier")
print("=" * 60)

# Check if test folder exists
if not os.path.exists(TEST_FOLDER):
    print(f"Error: '{TEST_FOLDER}' folder not found.")
    exit()

files = os.listdir(TEST_FOLDER)

if len(files) == 0:
    print("No images found in test_images folder.")
    exit()

# =====================================
# Predict Images
# =====================================
for filename in files:

    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        continue

    image_path = os.path.join(TEST_FOLDER, filename)

    try:
        # Load image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        # Normalize image (important if training used rescale=1./255)
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        print("\n" + "-" * 60)
        print(f"Image      : {filename}")

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"Prediction : {class_names[predicted_index]}")
            print(f"Confidence : {confidence * 100:.2f}%")
        else:
            print("Prediction : Unknown / Not Confident")
            print(f"Highest Confidence : {confidence * 100:.2f}%")

    except Exception as e:
        print(f"\nError processing {filename}")
        print(e)

print("\n" + "=" * 60)
print("Prediction Completed")
print("=" * 60)
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ==========================
# Load Model
# ==========================

MODEL_PATH = "model/animal_eye_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Class Names
# ==========================

class_names = [
    "Cat",
    "Dog",
    "Wild"
]

# ==========================
# Test Folder
# ==========================

TEST_FOLDER = "test_images"

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

print("=" * 60)
print("Animal Eye Classifier")
print("=" * 60)

if not os.path.exists(TEST_FOLDER):
    print("Test folder not found!")
    exit()

files = os.listdir(TEST_FOLDER)

for filename in files:

    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        continue

    image_path = os.path.join(TEST_FOLDER, filename)

    try:

        # Load image
        img = image.load_img(image_path, target_size=(224,224))

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        # EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = prediction[0][predicted_index]

        print("-"*60)
        print("Image :", filename)
        print("Prediction :", class_names[predicted_index])
        print(f"Confidence : {confidence*100:.2f}%")

        print("\nAll Class Probabilities")

        for i, cls in enumerate(class_names):
            print(f"{cls:5}: {prediction[0][i]*100:.2f}%")

        print("-"*60)

    except Exception as e:
        print("Error :", e)

print("\nPrediction Completed.")
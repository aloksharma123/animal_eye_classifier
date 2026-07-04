import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ======================
# Load trained model
# ======================

model = tf.keras.models.load_model("model/animal_eye_classifier.keras")

# Class names (use the same order as train_dataset.class_names)
class_names = [
    "Cat",
    "Dog",
    "Wild"
]

# Folder containing test images
TEST_FOLDER = "test_images"

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("=" * 60)
print("Animal Eye Classifier")
print("=" * 60)

# Loop through every image in the folder
# for filename in os.listdir(TEST_FOLDER):

   # Loop through every image in the folder
for filename in os.listdir(TEST_FOLDER):

    print("Found:", filename)

    if filename.lower().endswith(image_extensions):

        image_path = os.path.join(TEST_FOLDER, filename)

        # Load image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"\nImage      : {filename}")
        print(f"Prediction : {predicted_class}")
        print(f"Confidence : {confidence:.2f}%")
         
         
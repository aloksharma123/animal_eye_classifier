import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION
# ==========================

DATASET_PATH = "eye_dataset/train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ==========================
# LOAD DATASET
# ==========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nClasses:")
print(train_dataset.class_names)

# ==========================
# DATA AUGMENTATION
# ==========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ==========================
# PERFORMANCE
# ==========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

print("\nDataset loaded successfully.")

# ==========================
# LOAD PRETRAINED MODEL
# ==========================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained weights
base_model.trainable = False

# ==========================
# BUILD MODEL
# ==========================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu")(x)

outputs = layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ==========================
# COMPILE MODEL
# ==========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# TRAIN MODEL
# ==========================

EPOCHS = 15

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==========================
# SAVE MODEL
# ==========================

model.save("model/animal_eye_classifier.keras")

print("\nModel saved successfully!")

# ==========================
# PLOT RESULTS
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.show()
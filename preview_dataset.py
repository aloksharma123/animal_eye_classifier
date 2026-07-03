import os
import random
import cv2
import matplotlib.pyplot as plt

folder = "eye_dataset/train/cat"

images = random.sample(os.listdir(folder), 9)

plt.figure(figsize=(8,8))

for i, img_name in enumerate(images):
    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(img_name, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()
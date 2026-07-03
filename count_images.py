import os

base_folder = "eye_dataset/train"

classes = ["cat", "dog", "wild"]

total = 0

for animal in classes:
    folder = os.path.join(base_folder, animal)

    if os.path.exists(folder):
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    else:
        count = 0

    print(f"{animal.capitalize():<5}: {count} images")
    total += count

print("-" * 25)
print(f"Total : {total} images")
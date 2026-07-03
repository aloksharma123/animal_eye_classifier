import os
import cv2

# -----------------------------
# CONFIGURATION
# -----------------------------
SOURCE = "afhq/train"
DEST = "eye_dataset/train"

classes = ["cat", "dog", "wild"]

print("\nChoose class to annotate:")
print("1. Cat")
print("2. Dog")
print("3. Wild")

choice = input("\nEnter choice (1/2/3): ")

if choice == "1":
    animal = "cat"
elif choice == "2":
    animal = "dog"
elif choice == "3":
    animal = "wild"
else:
    print("Invalid choice!")
    exit()

input_folder = os.path.join(SOURCE, animal)
output_folder = os.path.join(DEST, animal)

os.makedirs(output_folder, exist_ok=True)

images = sorted(os.listdir(input_folder))

drawing = False
ix = iy = fx = fy = -1


def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y


processed = 0

for img_name in images:

    save_path = os.path.join(output_folder, img_name)

    # Skip already annotated images
    if os.path.exists(save_path):
        processed += 1
        continue

    path = os.path.join(input_folder, img_name)

    image = cv2.imread(path)

    if image is None:
        continue

    clone = image.copy()

    ix = iy = fx = fy = -1

    cv2.namedWindow("Crop Eye")
    cv2.setMouseCallback("Crop Eye", draw_rectangle)

    print(f"\nProgress: {processed + 1}/{len(images)}")
    print(f"Image: {img_name}")
    print("S = Save | N = Skip | Q = Quit")

    while True:

        temp = clone.copy()

        if ix != -1 and fx != -1:
            cv2.rectangle(temp, (ix, iy), (fx, fy), (0, 255, 0), 2)

        cv2.imshow("Crop Eye", temp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            if ix == -1 or fx == -1:
                print("Draw a rectangle first!")
                continue

            x1 = min(ix, fx)
            x2 = max(ix, fx)

            y1 = min(iy, fy)
            y2 = max(iy, fy)

            crop = clone[y1:y2, x1:x2]

            if crop.size == 0:
                print("Invalid crop!")
                continue

            cv2.imwrite(save_path, crop)

            processed += 1

            print("Saved:", save_path)

            break

        elif key == ord("n"):

            print("Skipped")

            break

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print("\nProgress saved automatically.")

            exit()

cv2.destroyAllWindows()

print("\nAll images for", animal, "completed!")
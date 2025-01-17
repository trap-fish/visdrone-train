import cv2
import matplotlib.pyplot as plt
from time import sleep

img_number = '1793'
trackid = '30'
# Paths to image and label file
image_path = f"/media/citi-ai/matthew/visdrone-train/datasets/okutama/train/Drone1/Morning/Extracted-Frames-1280x720/1.1.1/{img_number}.jpg"
label_path = "/media/citi-ai/matthew/visdrone-train/datasets/okutama/train/Labels/SingleActionTrackingLabels/3840x2160/1.1.1.txt"  # Update this to the correct label file path

# Original resolution and target resolution
original_width, original_height = 3840, 2160
target_width, target_height = 1280, 720

# Scaling factors
scale_x = target_width / original_width
scale_y = target_height / original_height

# Read the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read labels and plot bounding boxes
with open(label_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        data = line.split()
        if data[0] == trackid and data[5] == img_number:
            print(f"Trackid - {data[0]=='99'} \t frameid - {data[5] == '2271'}")
            sleep(4)
            print("Processing first bounding box...")
            x1, y1, x2, y2 = map(int, data[1:5])  # Extract bounding box coordinates
            label = data[9].strip('"')  # Extract the label
            
            # Resize the bounding box according to scaling factors
            x1_resized, y1_resized = int(x1 * scale_x), int(y1 * scale_y)
            x2_resized, y2_resized = int(x2 * scale_x), int(y2 * scale_y)
            
            # Draw the resized bounding box
            cv2.rectangle(image, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 2)
            cv2.putText(image, label, (x1_resized, y1_resized - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Save the resized image with bounding boxes
            output_path = f"./image-rs-{img_number}.jpg"  # Specify your desired output path
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            print(f"Image saved at {output_path}")

        else:
            continue

# Display the resized image with bounding boxes
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()


import cv2
import matplotlib.pyplot as plt
from time import sleep

# Paths
image_path = "/media/citi-ai/matthew/visdrone-train/datasets/okutama/train/Drone1/Morning/Extracted-Frames-1280x720/1.1.1/0.jpg"
label_path = "/media/citi-ai/matthew/visdrone-train/datasets/okutama/train/Labels/SingleActionLabels/3840x2160/1.1.1.txt"  # Update this to the correct label file path

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read labels and plot bounding boxes
with open(label_path, "r") as file:
    lines = file.readlines()
    i=0
    for line in lines:
        data = line.split()
        if i==0:
            print("here mf")
            x1, y1, x2, y2 = map(int, data[1:5])  # Extract bounding box coordinates
            label = data[9].strip('"')  # Extract the label
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            i=i+1
        else:
            break

# # Display the image with bounding boxes
# plt.figure(figsize=(10, 6))
# plt.imshow(image)
# plt.axis("off")
# plt.show()

output_path = "./image.jpg"  # Specify your desired output path
cv2.imwrite(output_path, image)
print(f"Image saved at {output_path}")

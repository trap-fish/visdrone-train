from ultralytics import YOLO

# Load a model
model = YOLO("./runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)


results = model("./datasets/sample.mov", show=True, save=True)
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
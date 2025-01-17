from ultralytics import YOLO, RTDETR
import os

# # Load a model
# model = YOLO("yolo11n.pt") # trained on coco data only

# model paths
yolo11n_visdrone_pth = "/media/citi-ai/matthew/visdrone-train/runs/detect/train/weights/best.pt"
yolo11n_pth = "yolo11n.pt"
yolo11s_pth = "yolo11s.pt"
rtdetr_pth = "rtdetr-l.pt"
#rtdetr_resnet50_pth = "/media/citi-ai/matthew/visdrone-train/rtdetr-resnet50.yaml" # resnet50 backbone

model_dir = {
    #"yolo11n-vd": {"type": "yolo", "path": yolo11n_visdrone_pth},
    "yolo11n": {"type": "yolo", "path": yolo11n_pth},
    "yolo11s": {"type": "yolo", "path": yolo11s_pth},
    "rtdetr-l": {"type": "rtdetr-l", "path": rtdetr_pth},
    #"rtdetr-res50": {"type": "rtdetr_r50", "path": rtdetr_pth}
}

# Define experiments: model, optimizer, and learning rate combinations
experiments = [
    {"optimizer": "Adam", "lr": 0.01},
    {"optimizer": "Adam", "lr": 0.001},
    {"optimizer": "Adam", "lr": 0.0005},
    {"optimizer": "SGD", "lr": 0.01},
    {"optimizer": "SGD", "lr": 0.001},
    {"optimizer": "SGD", "lr": 0.0005},
    # duplicate runs for training on another dataset
    #{"optimizer": "SGD", "lr": 0.001},
]

# Paths and parameters
data_path = "Okutama.yaml"  # Path to your dataset YAML file
epochs = 30  # Number of training epochs
imgsz = 640  # Image size for training
results_dir = "/media/citi-ai/matthew/visdrone-train/results/exp1_training_results"  # Directory to save results

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

for model_name, model_info in model_dir.items():
    # Train the model with the custom optimizer
    # model.train(data="Okutama.yaml", epochs=20, imgsz=640, optimizer='Adam', lr0=0.001,  momentum=0.9)
    print(f"\nTraining with model: {model_name}")
    model_type = model_info["type"]
    model_path = model_info["path"]

    # initialise model
    if model_type == "yolo":
        model = YOLO(model_path)
    elif model_type == "rtdetr-l":
        model = RTDETR(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Loop through experiments
    for i, exp in enumerate(experiments):
        print(f"\nStarting Experiment {i + 1}: {exp}")
        print(i)
        
        # Define unique experiment name for results
        exp_name = f"exp_{i + 1}_{model_name}_{exp['optimizer']}_lr{exp['lr']}"

        # if model_type == "rtdetr-l" and i > 5:
        #     data_path = "VisDrone.yaml"
        #     print("never never never")
        # else:
        #     continue

        # Train the model
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            optimizer=exp["optimizer"],
            lr0=exp["lr"],
            patience=10,
            project=results_dir,
            name=exp_name
        )

        print(f"Experiment {i + 1} completed! Results saved in {results_dir}/{exp_name}")

print("\nAll experiments completed!")

from ultralytics import RTDETR
import os

# # Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/media/citi-ai/matthew/visdrone-train/results/exp1_training_results/exp_1_rtdetr-l_SGD_lr0.001_Okutama_finetuned/weights/best.onnx")

project_dir = "/media/citi-ai/matthew/visdrone-train/results/exp2_rtdetr_results/"
data_path = "Okutama.yaml"
os.makedirs(project_dir, exist_ok=True)

# Evaluate on the test dataset
results = model.val(data="Okutama.yaml", split='test', project=project_dir)  # Use the test split
results_path = project_dir

# Save results to a file
output_file = os.path.join(results_path, f"test_results.txt")
with open(output_file, "w") as f:
    f.write(str(results))
    f.close()

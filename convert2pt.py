import torch
from ultralytics import RTDETR  # Replace with the actual import for your RT-DETR model

import torch
from transformers import RTDetrForObjectDetection

# Load the pretrained model
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
# Count and print the total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Define the path to save the model
save_path = "./rtdetr_r50vd.pt"

# Save the entire model
torch.save(model, save_path)

print(f"Model saved to {save_path}")


# # Load the PyTorch model
# pth_path = "/media/citi-ai/matthew/visdrone-train/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
# model = RTDETR()  # Initialize the model (use the actual class definition)
# model.load_state_dict(torch.load(pth_path, map_location="cpu"))  # Load weights

# # Set the model to evaluation mode
# model.eval()

# # Convert the model to TorchScript
# pt_path = "/media/citi-ai/matthew/visdrone-train/rtdetr_r18vd_dec3_6x_coco_from_paddle.pt"
# scripted_model = torch.jit.script(model)  # TorchScript conversion
# scripted_model.save(pt_path)

# print(f"Model has been successfully converted and saved at {pt_path}")

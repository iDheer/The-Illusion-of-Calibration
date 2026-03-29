import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2

# Import your model and dataset
from models import NetraModel
from dataset_loader import ChaksuDataset

def generate_attention_map(image_tensor, model, device):
    """
    Passes an image through the ViT, extracting the self-attention map 
    of the [CLS] token from the final transformer block.
    """
    model.eval()
    with torch.no_grad():
        # Get attention weights by passing output_attentions=True to the HF backbone
        outputs = model.backbone(image_tensor.unsqueeze(0).to(device), output_attentions=True)
        attentions = outputs.attentions  # Tuple of (layer_1_attn, ..., layer_n_attn)
        if not attentions:
            raise ValueError("Model backbone did not return attentions. Ensure HF model supports output_attentions=True.")
            
        last_layer_attention = attentions[-1]  # Shape: [1, num_heads, seq_len, seq_len]
        
        # Mean across all 16 attention heads
        attn_heads_mean = last_layer_attention.mean(dim=1)  # Shape: [1, seq_len, seq_len]
        
        # We only care about what the [CLS] token (index 0) is attending to (indices 1 to end)
        cls_attention = attn_heads_mean[0, 0, 1:]  # Shape: [num_patches]
        
        # For a 512x512 input with 16x16 patch size, we have 32x32 patches
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        cls_attention = cls_attention.reshape(grid_size, grid_size)
        
        # Normalize between 0 and 1
        cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
        return cls_attention.cpu().numpy()

def overlay_heatmap(img_tensor, attention_map):
    """
    Resizes the 32x32 attention map and overlays it on the 512x512 original image.
    """
    # Convert normalized image tensor back to displayable un-normalized numpy array
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # Resize attention map to match image dimensions
    heatmap = cv2.resize(attention_map, (img_np.shape[1], img_np.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    # Overlay overlay
    overlayed = 0.5 * img_np + 0.5 * heatmap
    overlayed = np.clip(overlayed, 0, 1)
    return img_np, heatmap, overlayed

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    SOURCE_MODEL_PATH = "/workspace/results_run7/Source_AIROGS/model.pth"
    ADAPTED_MODEL_PATH = "/workspace/results_run7/SFDA_Target/model.pth"
    DATA_DIR = "/workspace/data/chaksu" # Target dataset
    NUM_SAMPLES = 3 # How many images to visualize
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the Models
    print("Loading Source Model...")
    model_source = NetraModel(num_classes=2).to(device)
    if os.path.exists(SOURCE_MODEL_PATH):
        model_source.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location=device))
    else:
        print(f"WARNING: Source model {SOURCE_MODEL_PATH} not found.")
        
    print("Loading Adapted (Collapsed) Model...")
    model_adapted = NetraModel(num_classes=2).to(device)
    if os.path.exists(ADAPTED_MODEL_PATH):
        model_adapted.load_state_dict(torch.load(ADAPTED_MODEL_PATH, map_location=device))
    else:
        print(f"WARNING: Adapted model {ADAPTED_MODEL_PATH} not found.")

    # 2. Setup Data
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ChaksuDataset(root_dir=DATA_DIR, transform=transform)
    # Pick a few glaucoma images if possible (label == 1)
    glaucoma_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    sample_indices = glaucoma_indices[:NUM_SAMPLES] if glaucoma_indices else range(NUM_SAMPLES)
    
    # 3. Generate Visualizations
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, 4 * NUM_SAMPLES))
    fig.suptitle("Visualizing DINOv3 [CLS] Attention", fontsize=16)
    
    axes[0, 0].set_title("Original Image (Target: Glaucoma)")
    axes[0, 1].set_title("Zero-Shot Source Model\n(Attends to Pathologic Features)")
    axes[0, 2].set_title("SFDA Collapsed Model\n(Attends to Random Background/Style)")

    import traceback
    for i, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]
        
        try:
            # Generate maps
            attn_src = generate_attention_map(img_tensor, model_source, device)
            attn_adapt = generate_attention_map(img_tensor, model_adapted, device)
            
            # Overlay
            img_np, _, overlay_src = overlay_heatmap(img_tensor, attn_src)
            _, _, overlay_adapt = overlay_heatmap(img_tensor, attn_adapt)
            
            # Plot
            axes[i, 0].imshow(img_np)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(overlay_src)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay_adapt)
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"Error visualizing sample {i}: {e}")
            traceback.print_exc()

    plt.tight_layout()
    plt.savefig("Run-7/attention_maps_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved comparison heatmaps to Run-7/attention_maps_comparison.png")

if __name__ == "__main__":
    main()

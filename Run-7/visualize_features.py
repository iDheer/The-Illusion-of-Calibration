import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your existing models and data loaders
from models import NetraDINOv3
from dataset_loader import ChaksuDataset # assuming this exists or similar

def extract_features(model, dataloader, device):
    """
    Passes data through the model and extracts the penultimate features 
    (before the final classification layer) to see how the model groups them.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            # Assuming models.py NetraDINOv3 has a method or we can intercept
            # If NetraDINOv3 returns logits, we might need its backbone features directly
            features = model.backbone(images) 
            # flatten if necessary
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
                
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.vstack(all_features), np.concatenate(all_labels)

def plot_tsne(features, labels, title, save_path):
    print(f"Computing t-SNE for {title}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label="Label (0 = Normal, 1 = Glaucoma)")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def main():
    # ---------------------------------------------------------
    # CONFIGURATION (Update paths if they differ slightly)
    # ---------------------------------------------------------
    SOURCE_MODEL_PATH = "/workspace/results_run7/Source_AIROGS/model.pth"
    ADAPTED_MODEL_PATH = "/workspace/results_run7/SFDA_Target/model.pth"
    DATA_DIR = "/workspace/data/chaksu" # Target dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the Target Dataset (Chaksu)
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ChaksuDataset(root_dir=DATA_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Dataset loading error: {e}")
        print("Please ensure DATA_DIR points to the correct Chaksu path.")
        return

    # 2. Extract and Plot: ZERO-SHOT SOURCE MODEL
    print("\n--- Evaluating Zero-Shot Source Model ---")
    if os.path.exists(SOURCE_MODEL_PATH):
        model_source = NetraDINOv3(num_classes=2).to(device)
        model_source.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location=device))
        
        feats_src, labels_src = extract_features(model_source, dataloader, device)
        plot_tsne(feats_src, labels_src, "t-SNE: Zero-Shot Source Model Features (Good Separation)", "Run-7/tsne_source.png")
    else:
        print(f"Source model not found at {SOURCE_MODEL_PATH}")

    # 3. Extract and Plot: ADAPTED TARGET MODEL (The Collapsed One)
    print("\n--- Evaluating Adapted Model (Mode Collapse) ---")
    if os.path.exists(ADAPTED_MODEL_PATH):
        model_adapted = NetraDINOv3(num_classes=2).to(device)
        model_adapted.load_state_dict(torch.load(ADAPTED_MODEL_PATH, map_location=device))
        
        feats_tgt, labels_tgt = extract_features(model_adapted, dataloader, device)
        plot_tsne(feats_tgt, labels_tgt, "t-SNE: SFDA Adapted Model Features (Catastrophic Mode Collapse)", "Run-7/tsne_adapted.png")
    else:
        print(f"Adapted model not found at {ADAPTED_MODEL_PATH}")
        
    print("\nAll Done! Add tsne_source.png and tsne_adapted.png to the LaTeX paper.")

if __name__ == "__main__":
    main()

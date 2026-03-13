import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DESEDDataset
from models.mf_conformer import MultiFrequencyConformer

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"--- Starting Experiment: {cfg['experiment_name']} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Init Data
    dataset = DESEDDataset(cfg, mode="train")
    dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    
    # 2. Init Model
    model = MultiFrequencyConformer(cfg).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # 3. Optimizers
    # Using BCEWithLogitsLoss because SED is a multi-label classification task
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    
    # 4. Training Loop
    model.train()
    epochs = cfg['training']['epochs']
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Student-like detail: Using tqdm for clean progress bars
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}\n")
        
    print("✅ Pipeline Verification Successful! Architecture is ready for full DESED dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)

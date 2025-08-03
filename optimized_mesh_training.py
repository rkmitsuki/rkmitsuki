# ------------------- CONFIGURATION -------------------
main_dataset_dir = "mesh_dataset_main"
image_folder = "images"
node_folder = "nodes"

batch_size = 32  # Increased from 8
learning_rate = 1e-3  # Increased for faster convergence
num_epochs = 100  # Reduced from 200
patience = 15  # Reduced from 25
random_seed = 42
max_node_len = 4548
image_size = 256
gradient_clip = 1.0

# Simplified settings
use_checkpoint = False  # Disabled for speed
max_spatial_size = 64  # Increased from 32

assert max_node_len % 2 == 0, "max_node_len must be even"

# ------------------- IMPORTS -------------------------
import matplotlib
matplotlib.use('Agg')
import os
import random
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from torch.amp import GradScaler, autocast
import sys
import matplotlib.pyplot as plt

# ------------------- SEED -----------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(random_seed)

# ------------------- LOGGER ---------------------------
class Logger:
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ------------------- DATASET --------------------------
class SimpleMeshDataset(Dataset):
    def __init__(self, image_dir, node_dir, transform=None, max_node_len=None):
        self.image_dir = image_dir
        self.node_dir = node_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform
        self.max_node_len = max_node_len
        
        # Pre-compute transforms for speed
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_id = os.path.splitext(image_name)[0].replace("_image", "")
        img_path = os.path.join(self.image_dir, image_name)
        node_path = os.path.join(self.node_dir, base_id + "_nodes.csv")

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = self.base_transform(image)
        
        # Load nodes
        node_data = np.loadtxt(node_path, delimiter=",", dtype=np.float32)
        if node_data.ndim > 1:
            node_data = node_data.flatten()

        length = len(node_data)
        if length % 2 != 0:
            node_data = node_data[:-1]
            length -= 1

        # Scale to pixel coordinates
        node_data = node_data * 100 + 128

        # Pad or truncate
        if length < self.max_node_len:
            node_data = np.pad(node_data, (0, self.max_node_len - length), mode='constant', constant_values=0)
        else:
            node_data = node_data[:self.max_node_len]
            length = self.max_node_len

        return image.float(), torch.tensor(node_data, dtype=torch.float32), length

# ------------------- SIMPLE MODEL ---------------------
class SimpleMeshPredictor(nn.Module):
    def __init__(self, num_outputs, hidden_dim=512):
        super().__init__()
        self.num_outputs = num_outputs
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Calculate feature size
        self.feature_size = 512 * 8 * 8
        
        # Simple MLP head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        
        # Ensure output is in reasonable range
        output = torch.clamp(output, min=0, max=256)
        
        return output

# ------------------- SIMPLE LOSS ----------------------
class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, lengths):
        """Simple MSE loss with length masking"""
        total_loss = 0
        valid_samples = 0
        
        for i, length in enumerate(lengths):
            if length > 0:
                pred_coords = pred[i, :length]
                target_coords = target[i, :length]
                
                loss = F.mse_loss(pred_coords, target_coords)
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)

# ------------------- METRICS --------------------------
def compute_simple_metrics(pred, target, lengths):
    """Compute simple metrics without Hungarian matching"""
    total_mse = 0
    total_mae = 0
    valid_samples = 0
    
    for i, length in enumerate(lengths):
        if length > 0:
            pred_coords = pred[i, :length]
            target_coords = target[i, :length]
            
            mse = F.mse_loss(pred_coords, target_coords).item()
            mae = F.l1_loss(pred_coords, target_coords).item()
            
            total_mse += mse
            total_mae += mae
            valid_samples += 1
    
    return {
        'mse': total_mse / max(valid_samples, 1),
        'mae': total_mae / max(valid_samples, 1)
    }

# ------------------- TRAINING FUNCTIONS ---------------
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    all_metrics = []
    
    for images, targets, lengths in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            predictions = model(images)
            loss = criterion(predictions, targets, lengths)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_simple_metrics(predictions, targets, lengths)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_metrics['loss'] = total_loss / len(loader.dataset)
    return avg_metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for images, targets, lengths in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                predictions = model(images)
                loss = criterion(predictions, targets, lengths)
            
            total_loss += loss.item() * images.size(0)
            
            metrics = compute_simple_metrics(predictions, targets, lengths)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_metrics['loss'] = total_loss / len(loader.dataset)
    return avg_metrics

# ------------------- VISUALIZATION --------------------
def visualize_simple_predictions(image_tensor, gt_coords, pred_coords, save_path, title=""):
    """Simple visualization without attention maps"""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    
    # Original predictions
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    
    # Reshape coordinates
    gt_coords = gt_coords.reshape(-1, 2)
    pred_coords = pred_coords.reshape(-1, 2)
    
    plt.scatter(gt_coords[:, 0], gt_coords[:, 1], c='lime', s=20, label='GT', alpha=0.7)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', s=20, label='Pred', alpha=0.7)
    plt.title(f"{title} - Predictions")
    plt.legend()
    plt.axis('off')
    
    # Error visualization
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    for i in range(min(len(gt_coords), len(pred_coords))):
        plt.plot([gt_coords[i, 0], pred_coords[i, 0]], 
                [gt_coords[i, 1], pred_coords[i, 1]], 
                'yellow', alpha=0.5, linewidth=1)
    plt.title("Prediction Errors")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

# ------------------- MAIN TRAINING LOOP ---------------
if __name__ == "__main__":
    
    # Setup logging
    log_filename = f"simple_training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    sys.stdout = Logger(log_filename)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("SIMPLIFIED MESH NODE PREDICTION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Architecture: Simple CNN + MLP")
    print("="*60 + "\n")
    
    # Create dataset
    dataset = SimpleMeshDataset(
        os.path.join(main_dataset_dir, image_folder),
        os.path.join(main_dataset_dir, node_folder),
        max_node_len=max_node_len
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Simple train/val split instead of k-fold for speed
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Increased
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4  # Increased
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create simple model
    print("\nInitializing model...")
    model = SimpleMeshPredictor(num_outputs=max_node_len, hidden_dim=512).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Simple loss function
    criterion = SimpleMSELoss()
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{num_epochs} (Time: {epoch_time:.1f}s)")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | MSE: {train_metrics['mse']:.4f} | MAE: {train_metrics['mae']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | MSE: {val_metrics['mse']:.4f} | MAE: {val_metrics['mae']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Visualize predictions every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                for i, (img, target, length) in enumerate(val_loader):
                    if i >= 1:  # Just first batch
                        break
                    
                    img = img.to(device)
                    pred = model(img)
                    
                    # Get first sample
                    pred_coords = pred[0, :length[0]].cpu().numpy()
                    target_coords = target[0, :length[0]].cpu().numpy()
                    
                    save_path = f"viz_outputs/epoch_{epoch}_sample.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    visualize_simple_predictions(
                        img[0].cpu(),
                        target_coords,
                        pred_coords,
                        save_path,
                        title=f"Epoch {epoch+1}"
                    )
            model.train()
        
        # Early stopping
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            patience_counter = 0
            print(f"✓ New best model! Loss: {best_loss:.4f}")
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss,
                'metrics': val_metrics
            }, 'checkpoints/best_simple_model.pth')
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
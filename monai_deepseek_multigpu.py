import sys
import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet, BasicUNet
from monai.transforms import Compose, ToTensord
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Fix typo in original code: 'Paraser' -> 'ArgumentParser'
parser = argparse.ArgumentParser(description='Train a unetr/swin_unetr model')
parser.add_argument('--model', type=str, default='unetr', 
                    help='Model name, options: unet, attention_unet, unetr, swin_unetr (default: unetr)')
parser.add_argument('--loss', type=str, default='mse', 
                    help='Loss function, options: mse, mae, (default: mse)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='Data directory path (default: /data/)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size per gpu for training (default: 1)')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs (default: 10)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of cpus for preprocessing (default: 4)')
parser.add_argument('--is_log', type=bool, default=False,
                    help='whether to log the information in tensorboard (default: False)')
args = parser.parse_args()

# Check for available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs detected: {num_gpus}")

# Configuration
data_dir = args.data_dir
DATA_DIRS = {
    'train': os.path.join(data_dir, 'train'),
    'val': os.path.join(data_dir, 'validation'),
    'test': os.path.join(data_dir, 'test')
}
LOSS_FUNCTION = args.loss
BATCH_SIZE = args.batch_size * num_gpus
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
MODEL_NAME = args.model
NUM_WORKERS = args.num_workers
IS_LOG = args.is_log
VAL_INTERVAL = 2
IMG_SIZE = (128, 128, 128) #(192, 192, 192)

"""
print("Configuration:")
print(f"Model: {args.model}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Epochs: {args.epochs}")
print(f"Batch Size: {args.batch_size}")
print(f"Number of CPU workers: {args.num_workers}")
"""

# tensorboard logs
if IS_LOG:
    writer = SummaryWriter(log_dir='tensorboard_logs/')

# Custom Dataset Class (remain the same)
class MatDataset(Dataset):
    def __init__(self, file_groups, transforms=None):
        self.file_groups = file_groups
        self.transforms = transforms
        
    def __len__(self):
        return len(self.file_groups)
    
    def __getitem__(self, index):
        ct_path, jaw_path, mlc_path, dose_path = self.file_groups[index]
        
        ct = loadmat(ct_path)['array'].astype(np.float32)
        jaw = loadmat(jaw_path)['array'].astype(np.float32)
        mlc = loadmat(mlc_path)['array'].astype(np.float32)
        
        image = np.stack([ct, jaw, mlc], axis=0)
        label = loadmat(dose_path)['array'].astype(np.float32)[np.newaxis, ...]
        
        sample = {'image': image, 'label': label}
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

# Create file groups (remain the same)
def create_file_groups(data_dir):
    ct_files = sorted(glob.glob(os.path.join(data_dir, 'CT_*.mat')))
    file_groups = []
    
    for ct_path in ct_files:
        base_name = os.path.basename(ct_path)
        identifier = '_'.join(base_name.split('_')[1:])
        
        jaw_path = os.path.join(data_dir, f'jaw_{identifier}')
        mlc_path = os.path.join(data_dir, f'mlc_{identifier}')
        dose_path = os.path.join(data_dir, f'Dose_{identifier}')
        
        if all(os.path.exists(p) for p in [jaw_path, mlc_path, dose_path]):
            file_groups.append((ct_path, jaw_path, mlc_path, dose_path))
    
    return file_groups

def main():
    # Create datasets
    basic_transforms = Compose([ToTensord(keys=['image', 'label'])])
    
    train_groups = create_file_groups(DATA_DIRS['train'])
    val_groups = create_file_groups(DATA_DIRS['val'])
    test_groups = create_file_groups(DATA_DIRS['test'])

    train_ds = MatDataset(train_groups, basic_transforms)
    val_ds = MatDataset(val_groups, basic_transforms)
    test_ds = MatDataset(test_groups, basic_transforms)

    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Initialize model with DataParallel
    model_class = {
        'swin_unetr': SwinUNETR,
        'unetr': UNETR,
        'attention_unet': AttentionUnet,
        'unet': BasicUNet
    }[MODEL_NAME]

    model_kwargs = {
        'swin_unetr': {
            'in_channels': 3,
            'out_channels': 1,
            'img_size': IMG_SIZE,
            'feature_size': 24,
            'depths': (2, 2, 2, 2),
            'num_heads': (3, 6, 12, 24),
            'norm_name': "instance",
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'dropout_path_rate': 0.0,
            'normalize': True,
            'spatial_dims': 3,
            'use_v2': False
        },
        'unetr': {
            'in_channels': 3,
            'out_channels': 1,
            'img_size': IMG_SIZE,
            'feature_size': 16,
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'proj_type': "conv",
            'norm_name': "instance",
            'conv_block': True,
            'res_block': True,
            'dropout_rate': 0.0,
            'spatial_dims': 3
        },
        'attention_unet': {
            'spatial_dims': 3,
            'in_channels': 3,
            'out_channels': 1,
            'channels': (64, 128, 256, 512, 1024),
            'strides': (2, 2, 2, 2)
        },
        'unet': {
            'spatial_dims': 3,
            'in_channels': 3,
            'out_channels': 1
        }
    }[MODEL_NAME]

    model = model_class(**model_kwargs).to(device)
    
    # Wrap model with DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Loss and optimizer
    if LOSS_FUNCTION == 'mse':
        loss_function = torch.nn.MSELoss() 
    elif LOSS_FUNCTION == "mae":
        loss_function = torch.nn.L1Loss()
    else:
        print("loss function incorrectly defined", LOSS_FUNCTION)
        sys.exit()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    train_loss_values = []
    val_loss_values = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_loss_values.append(epoch_loss)
        print(f"Train loss: {epoch_loss:.4f}")
        
        if IS_LOG:
            writer.add_scalars('Loss/Train', {'Loss':epoch_loss}, epoch)
        
        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch['image'].to(device, non_blocking=True)
                    val_labels = val_batch['label'].to(device, non_blocking=True)
                    
                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_labels).item()
            
            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.4f}")
            if IS_LOG:
                writer.add_scalars('Loss/Validation', {'Loss':val_loss}, epoch)
            
            # Save best model (handling DataParallel)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), 'best_model.pth')
                else:
                    torch.save(model.state_dict(), 'best_model.pth')
                print("Saved new best model")

    # Plot training curves (remain the same)
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(np.arange(1, len(val_loss_values)+1)*VAL_INTERVAL-1, val_loss_values, label='Validation Loss')
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_validation_loss.png', format='png', dpi=200, bbox_inches='tight')

    if IS_LOG:
        writer.close()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
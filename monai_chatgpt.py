import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss  # or torch.nn.L1Loss for MAE
from scipy.io import loadmat

# MONAI
import monai
from monai.transforms import Compose, ToTensorD
from monai.networks.nets import UNETR, SwinUNETR  # for "swin_unet" option


# -------------------------------------------------------------------
# 1) Function to discover valid sample keys in a directory
#    by checking for CT_*.mat, jaw_*.mat, mlc_*.mat, Dose_*.mat
# -------------------------------------------------------------------
def get_valid_sample_keys(data_dir, mat_key="vol"):
    """
    Scan `data_dir` for files matching 'CT_*.mat'. For each found,
    parse out the key (xx_yy_zz) and verify that
    'jaw_{key}.mat', 'mlc_{key}.mat', and 'Dose_{key}.mat' exist.
    
    Returns:
        valid_keys (list[str]): list of the valid sample keys found.
    """
    pattern_ct = os.path.join(data_dir, "CT_*.mat")
    ct_files = glob.glob(pattern_ct)

    valid_keys = []
    for ct_path in ct_files:
        filename = os.path.basename(ct_path)       # e.g. "CT_01_02_03.mat"
        key_str = filename.replace("CT_", "")      # "01_02_03.mat"
        key_str = key_str.replace(".mat", "")      # "01_02_03"

        jaw_path  = os.path.join(data_dir, f"jaw_{key_str}.mat")
        mlc_path  = os.path.join(data_dir, f"mlc_{key_str}.mat")
        dose_path = os.path.join(data_dir, f"Dose_{key_str}.mat")

        if all(os.path.exists(p) for p in [jaw_path, mlc_path, dose_path]):
            valid_keys.append(key_str)

    return valid_keys


# -------------------------------------------------------------------
# 2) Custom Dataset for loading 3-channel inputs + 1-channel output
# -------------------------------------------------------------------
class RadiotherapyDoseDataset(Dataset):
    """
    Loads 3 channels from .mat files (CT, jaw, mlc)
    and a continuous 'Dose' volume from .mat files.
    
    Each volume is expected to be shape [192, 192, 192].
    We stack the 3 inputs to form [3, 192, 192, 192],
    and expand the dose to [1, 192, 192, 192].
    """
    def __init__(self, sample_keys, data_dir, transform=None, mat_key="vol"):
        """
        Args:
            sample_keys (list[str]): list of ID strings matching xx_yy_zz
            data_dir (str): directory containing the .mat files
            transform (callable): optional MONAI transforms (dict-based)
            mat_key (str): variable name inside .mat file
        """
        self.sample_keys = sample_keys
        self.data_dir = data_dir
        self.transform = transform
        self.mat_key = mat_key

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]

        # Construct file paths
        ct_path   = os.path.join(self.data_dir, f"CT_{key}.mat")
        jaw_path  = os.path.join(self.data_dir, f"jaw_{key}.mat")
        mlc_path  = os.path.join(self.data_dir, f"mlc_{key}.mat")
        dose_path = os.path.join(self.data_dir, f"Dose_{key}.mat")

        # Load each channel and convert to float32
        ct_data   = loadmat(ct_path)[self.mat_key].astype(np.float32)
        jaw_data  = loadmat(jaw_path)[self.mat_key].astype(np.float32)
        mlc_data  = loadmat(mlc_path)[self.mat_key].astype(np.float32)

        # Stack into [3, 192, 192, 192]
        image_array = np.stack([ct_data, jaw_data, mlc_data], axis=0)

        # Load dose [192, 192, 192] -> expand to [1, 192, 192, 192]
        dose_data = loadmat(dose_path)[self.mat_key].astype(np.float32)
        dose_data = np.expand_dims(dose_data, axis=0)

        sample = {
            "image": image_array,
            "label": dose_data
        }

        # Apply MONAI-style transforms (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]


# -------------------------------------------------------------------
# 3) Helper function to create model (UNETR or SwinUNETR)
# -------------------------------------------------------------------
def create_model(model_name="unetr", in_channels=3, out_channels=1, img_size=(192, 192, 192)):
    """
    Create a model (UNETR or SwinUNETR) for 3D volumes.

    Args:
        model_name: 'unetr' or 'swin_unet'
        in_channels: number of input channels (3 if CT/jaw/mlc)
        out_channels: number of output channels (1 for Dose)
        img_size: tuple of (H, W, D)
    """
    if model_name.lower() == "unetr":
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0
        )
    elif model_name.lower() in ["swin_unet", "swin_unetr"]:
        # MONAI's class is SwinUNETR, often called Swin UNet
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=48,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' is not recognized. Use 'unetr' or 'swin_unet'.")

    return model


# -------------------------------------------------------------------
# 4) Main Script: Uses train/validation/test directories
# -------------------------------------------------------------------
def main():
    # Adjust these paths/parameters as needed
    train_dir = "/path/to/train"         # directory containing CT_xx_yy_zz.mat, ...
    val_dir   = "/path/to/validation"    # directory for validation data
    test_dir  = "/path/to/test"          # directory for test data
    mat_key   = "vol"                    # key inside your .mat file
    model_name = "unetr"                 # can be "unetr" or "swin_unet"
    batch_size = 1
    max_epochs = 10
    val_interval = 2  # validate every N epochs

    # --- 4.1) Discover sample keys in each folder ---
    train_keys = get_valid_sample_keys(train_dir, mat_key=mat_key)
    val_keys   = get_valid_sample_keys(val_dir,   mat_key=mat_key)
    test_keys  = get_valid_sample_keys(test_dir,  mat_key=mat_key)

    print(f"Found {len(train_keys)} samples in train directory: {train_dir}")
    print(f"Found {len(val_keys)} samples in validation directory: {val_dir}")
    print(f"Found {len(test_keys)} samples in test directory: {test_dir}")

    # --- 4.2) Create datasets ---
    basic_transforms = Compose([ToTensorD(keys=["image", "label"])])

    train_dataset = RadiotherapyDoseDataset(
        sample_keys=train_keys,
        data_dir=train_dir,
        transform=basic_transforms,
        mat_key=mat_key
    )
    val_dataset = RadiotherapyDoseDataset(
        sample_keys=val_keys,
        data_dir=val_dir,
        transform=basic_transforms,
        mat_key=mat_key
    )
    test_dataset = RadiotherapyDoseDataset(
        sample_keys=test_keys,
        data_dir=test_dir,
        transform=basic_transforms,
        mat_key=mat_key
    )

    # --- 4.3) Create data loaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # --- 4.4) Create model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=model_name, 
        in_channels=3,          # (CT, jaw, mlc)
        out_channels=1,         # single channel for Dose
        img_size=(192, 192, 192)
    ).to(device)

    # --- 4.5) Loss and optimizer ---
    # For regression tasks (dose), MSELoss or L1Loss is typical
    criterion = MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 4.6) Training and validation loop ---
    for epoch in range(max_epochs):
        print(f"Epoch [{epoch+1}/{max_epochs}]")
        
        # ---------------------------- TRAIN ----------------------------
        model.train()
        train_loss_epoch = 0.0
        num_train_steps = 0
        
        for batch_data in train_loader:
            num_train_steps += 1
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)     # [B, 1, 192, 192, 192]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= num_train_steps
        print(f"  Train MSE: {train_loss_epoch:.4f}")

        # ---------------------------- VALIDATION ----------------------------
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss_epoch = 0.0
            num_val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    num_val_steps += 1
                    val_images, val_labels = val_data
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_images)
                    loss = criterion(val_outputs, val_labels)
                    val_loss_epoch += loss.item()

            val_loss_epoch /= num_val_steps
            print(f"  Validation MSE: {val_loss_epoch:.4f}")

    # --- 4.7) Final Test Evaluation ---
    print("Evaluating on the test set...")
    model.eval()
    test_loss = 0.0
    test_steps = 0

    with torch.no_grad():
        for test_data in test_loader:
            test_steps += 1
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_images)
            loss = criterion(test_outputs, test_labels)
            test_loss += loss.item()

    test_loss /= test_steps if test_steps > 0 else 1
    print(f"Final Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()
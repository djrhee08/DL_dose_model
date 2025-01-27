import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader, TensorDataset

def test_unetr():
    # Define device: use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy input data
    # Shape: (batch_size, in_channels, D, H, W)
    batch_size = 2
    in_channels = 3
    out_channels = 1
    img_size = (96, 96, 96)
    num_samples = 10

    # Initialize the UNETR model with specified parameters
    net = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=img_size, feature_size=32, norm_name='instance').to(device)

    print("UNETR model initialized successfully.")

    dummy_input = torch.randn(batch_size, in_channels, *img_size).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Perform a forward pass
    with torch.no_grad():
        output = net(dummy_input)
    
    print(f"Output shape: {output.shape}")

    # Verify output dimensions
    expected_shape = (batch_size, out_channels, *img_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print("UNETR forward pass successful. Output shape is as expected.")


    # Generate random input data
    inputs = torch.randn(num_samples, in_channels, *img_size)
    # Generate random target data as integer labels for segmentation
    # Each voxel label is between 0 and out_channels - 1
    targets = torch.randint(0, out_channels, (num_samples, *img_size), dtype=torch.long)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dummy dataset created with {num_samples} samples.")

    # 4. Define Loss Function and Optimizer
    # Using CrossEntropyLoss which combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    print("Loss function and optimizer set up.")

    # 5. Training Loop
    num_epochs = 5
    net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = net(data)  # Output shape: (batch_size, out_channels, D, H, W)

            # Compute loss
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

    print("Training with dummy data completed successfully.")

    # 6. (Optional) Validation with Dice Metric
    # Note: Since the data is random, the Dice score is not meaningful here.
    # This is just to demonstrate how to compute metrics.
    net.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            outputs = net(data)
            # Get the probabilities by applying softmax
            probs = torch.softmax(outputs, dim=1)
            # Get the predicted class by selecting the max probability
            preds = torch.argmax(probs, dim=1)
            dice_metric(y_pred=preds, y=target)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Dummy Dice Score: {dice_score:.4f}")


if __name__ == "__main__":
    test_unetr()


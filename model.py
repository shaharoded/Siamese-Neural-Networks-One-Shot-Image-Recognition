import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader, stratified_split

MODEL_DIR = "trained_models"


class CNNBlock(nn.Module):
    """
    A single convolutional block consisting of:
    - Convolutional layer
    - ReLU activation
    - Optional MaxPooling layer

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (filters).
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to the input.
        use_pooling (bool): Whether to include a max-pooling layer.

    Forward Pass:
        Input: Tensor of shape (batch_size, in_channels, height, width)
        Output: Processed tensor after convolution, ReLU, and optional pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding, use_pooling=True, use_batchnorm=True, dropout_prob=0.0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pooling else None
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class SiameseNetwork(nn.Module):
    """
    A Siamese Neural Network for comparing two images, designed for one-shot image recognition.

    Architecture:
    - Two identical branches (twin networks) built from a sequence of CNN blocks.
    - Fully connected layers to extract feature embeddings.
    - An L1 distance layer to compare feature embeddings from both branches.
    - A sigmoid layer to output a similarity score (0 = dissimilar, 1 = similar).

    Args:
        cnn_blocks (list): List of dictionaries defining the CNN blocks.
                          Each dictionary includes:
                          - 'out_channels', 'kernel_size', 'stride', 'padding', 'use_pooling'.
        fc_layers (list): List of dictionaries defining the fully connected layers.
                          Each dictionary includes:
                          - 'in_features', 'out_features'.

    Forward Pass:
        Input: Two images (input1, input2) of shape (batch_size, channels, height, width).
        Output: Similarity score between 0 and 1.
    """
    def __init__(self, cnn_blocks, fc_layers):
        super(SiameseNetwork, self).__init__()
        # Dynamically create CNN blocks
        cnn_layers = []
        in_channels = 1  # Assuming grayscale input; update if RGB
        for block in cnn_blocks:
            cnn_layers.append(CNNBlock(
                in_channels=in_channels,
                out_channels=block["out_channels"],
                kernel_size=block["kernel_size"],
                stride=block["stride"],
                padding=block["padding"],
                use_pooling=block["use_pooling"]
            ))
            in_channels = block["out_channels"]
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Fully connected layers
        fc_layers_list = []
        input_features = fc_layers[0]["in_features"]
        for fc in fc_layers:
            fc_layers_list.append(nn.Linear(input_features, fc["out_features"]))
            fc_layers_list.append(nn.ReLU())
            input_features = fc["out_features"]
        self.fc = nn.Sequential(*fc_layers_list)
        
        # Output similarity measure
        self.out = nn.Linear(fc_layers[-1]["out_features"], 1)

    def forward_once(self, x):
        """
        Passes a single input through the CNN and fully connected layers
        to extract its feature vector.

        Input: Tensor of shape (batch_size, channels, height, width)
        Output: Feature vector of shape (batch_size, feature_dim)
        """
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        """
        Processes two inputs (input1, input2) and computes their similarity score.

        Steps:
        1. Extract feature vectors for both inputs using the twin networks.
        2. Compute the L1 distance between the two feature vectors.
        3. Pass the L1 distance through a fully connected layer with a sigmoid to
           output the similarity score.

        Input: Two tensors, each of shape (batch_size, channels, height, width)
        Output: Tensor of shape (batch_size, 1) representing similarity scores.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Compute L1 distance and pass through a sigmoid
        l1_distance = torch.abs(output1 - output2)
        return torch.sigmoid(self.out(l1_distance))
    

def train(model, dataset, batch_size, shuffle, val_split, 
          epochs, lr, l2_reg, early_stop_patience, save_path, num_workers, device):
    """
    Train the Siamese Neural Network.
        Args:
        model (SiameseNetwork): The Siamese Neural Network model.
        dataset (SiameseNetworkDataset): Dataset object for training.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset during training.
        val_split (float): Fraction of data to use for validation.
        epochs (int): Maximum number of epochs to train.
        lr (float): Learning rate for the optimizer.
        l2_reg (float): L2 regularization strength.
        early_stop_patience (int): Number of epochs to wait for improvement before stopping.
        save_path (str): Path to save the best model.
        num_workers (int): Number of worker threads for data loading.
        device (str): Device to run the training on (e.g., 'cpu', 'cuda').

    Returns:
        2 list objects - train and validation losses.
        In addition, a trained model's state dict will be saved in save_path
    """
    # Record start time
    start_time = time.time()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Perform stratified split
    train_dataset, val_dataset = stratified_split(dataset, val_split)

    # Create DataLoaders
    train_loader = get_dataloader(train_dataset, batch_size, shuffle, num_workers)
    val_loader = get_dataloader(val_dataset, batch_size, False, num_workers)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img1, img2, labels = batch
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_end_time = time.time()
        current_duration = (epoch_end_time - start_time)/60

        print(f"[Training Status]: Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {current_duration:.2f} minutes")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model
            torch.save(model.state_dict(), os.path.join(model_dir, save_path))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"[Training Status]: Early stopping after {epoch+1} epochs.")
            break

    print(f"[Training Status]: Best model saved at {save_path} with Validation Loss: {best_val_loss:.4f}")
    return train_losses, val_losses


def predict(model_path, test_dataset, batch_size, num_workers, device):
    """
    Predict using a trained Siamese Neural Network on a test dataset.

    Args:
        model_path (str): The file name fo the model, within MODEL_DIR.
        test_dataset (SiameseNetworkDataset): Test dataset object.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
        device (str): Device to run the predictions on (e.g., 'cpu', 'cuda').

    Returns:
        float: Accuracy of the model on the test dataset.
    """
    # Check if model exists
    model_path = os.path.join(MODEL_DIR, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Error]: The specified model path '{model_path}' does not exist.")
    
    # Load the trained model
    print(f"[Prediction]: Loading model from {model_path}...")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create DataLoader for test dataset
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct = 0
    total = 0

    # Predict on the test dataset
    print("[Prediction]: Starting predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2).squeeze()
            predictions = (outputs > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"[Prediction]: Accuracy: {accuracy:.4f}")
    return accuracy
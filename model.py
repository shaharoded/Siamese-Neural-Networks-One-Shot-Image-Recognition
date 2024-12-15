import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from dataset import get_dataloader, stratified_split, augment_dataset, count_labels


SEED = 42

def set_seed(seed):
    '''
    Fix all randomized actions
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


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
        
        # Define layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pooling else None
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        
        # Initialize weights and biases
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-2)
        if self.conv.bias is not None:
            nn.init.normal_(self.conv.bias, mean=0.5, std=1e-2)

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
    
    
class FullyConnectedBlock(nn.Module):
    """
    A block for creating fully connected layers with optional batch normalization and dropout.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        use_batchnorm (bool): Whether to apply batch normalization.
        dropout_prob (float): Dropout probability (0.0 means no dropout).

    Forward Pass:
        Input: Tensor of shape (batch_size, in_features)
        Output: Tensor of shape (batch_size, out_features)
    """
    def __init__(self, in_features, out_features, use_batchnorm=False, dropout_prob=0.0):
        super(FullyConnectedBlock, self).__init__()
        
        # Define layers
        self.linear = nn.Linear(in_features, out_features, bias=not use_batchnorm)
        self.batchnorm = nn.BatchNorm1d(out_features) if use_batchnorm else None
        self.activation = nn.Sigmoid()  # Using Sigmoid for all FC layers
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        
        # Initialize weights
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.2)
        if self.linear.bias is not None:
            nn.init.normal_(self.linear.bias, mean=0.5, std=1e-2)


    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
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
            fc_layers_list.append(FullyConnectedBlock(
                in_features=input_features,
                out_features=fc["out_features"],
                use_batchnorm=fc.get("use_batchnorm", False),
                dropout_prob=fc.get("dropout_prob", 0.0)
            ))
            input_features = fc["out_features"]
        self.fc = nn.Sequential(*fc_layers_list)
        
        
    def __forward_once(self, x):
        """
        Passes a single input through the CNN and the first fully connected layer
        to extract its 4096-dimensional feature vector.

        Input: Tensor of shape (batch_size, channels, height, width)
        Output: Feature vector of shape (batch_size, 4096)
        """
        x = self.cnn(x)  # Pass through CNN layers
        x = torch.flatten(x, 1)  # Flatten the tensor for fully connected layers
        x = self.fc[:-1](x)  # Pass through all FC layers except the last one
        return x

    def forward(self, input1, input2):
        """
        Processes two inputs (input1, input2) and computes their similarity score.

        Steps:
        1. Extract 4096-dimensional feature vectors for both inputs using `__forward_once`.
        2. Compute the L1 distance between the two feature vectors.
        3. Pass the L1 distance through the final fully connected layer to produce the similarity score.

        Input: Two tensors, each of shape (batch_size, channels, height, width)
        Output: Tensor of shape (batch_size, 1) representing similarity scores.
        """
        # Extract feature vectors for both inputs
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)

        # Compute L1 distance
        l1_distance = torch.abs(output1 - output2)

        # Pass the L1 distance through the final layer (last FC)
        return self.fc[-1](l1_distance)
    

def train(model, dataset, batch_size, val_split, augment_ratio,
          epochs, lr, l2_reg, early_stop_patience, save_path, num_workers, device):
    """
    Train the Siamese Neural Network.
        Args:
        model (SiameseNetwork): The Siamese Neural Network model.
        dataset (SiameseNetworkDataset): Dataset object for training.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset during training.
        val_split (float): Fraction of data to use for validation.
        augment_ratio (int): The ratio to augment the train dataset (X2, X3...)
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
    
    # Perform stratified split
    train_dataset, val_dataset = stratified_split(dataset, val_split)
    
    # Apply augmentation to the train dataset
    if augment_ratio > 0:
        print(f"[Training]: Applying augmentation with ratio {augment_ratio} to the dataset.")
        augmented_data = augment_dataset(train_dataset, augment_ratio)
        train_dataset.data = augmented_data
        
    train_positives, train_negatives = count_labels(train_dataset)
    val_positives, val_negatives = count_labels(val_dataset)
    
    print(f"[Data Distribution]: Subset Train - Positives: {train_positives}, Negatives: {train_negatives}")
    print(f"[Data Distribution]: Validation - Positives: {val_positives}, Negatives: {val_negatives}")

    # Create DataLoaders
    train_loader = get_dataloader(train_dataset, batch_size, num_workers)
    val_loader = get_dataloader(val_dataset, batch_size, num_workers)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Initialize variables for early stopping
    best_val_loss = np.inf
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.round() == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                img1, img2, labels = batch
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.round() == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        epoch_end_time = time.time()
        current_duration = (epoch_end_time - start_time)/60

        print(f"[Training Status]: Epoch {epoch+1}, Train Loss: {train_loss:.4f} (acc {train_accuracy:.3f}), Val Loss: {val_loss:.4f} (acc {val_accuracy:.3f}), Time: {current_duration:.1f} min, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping check, by validation accuracy check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"[Training Status]: Early stopping after {epoch+1} epochs.")
            break

    print(f"[Training Status]: Best model saved at {save_path} with Validation Loss: {best_val_loss:.4f}")
    return train_losses, val_losses


def predict(model_path, cnn_blocks, fc_layers, test_dataset, batch_size, num_workers, device):
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Error]: The specified model path '{model_path}' does not exist.")
    
    # Load the trained model
    print(f"[Prediction]: Loading model from {model_path}...")
    model = SiameseNetwork(cnn_blocks, fc_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Create DataLoader for test dataset
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    correct = 0
    total = 0

    # Predict on the test dataset
    print("[Prediction]: Starting predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Model output
            outputs = model(img1, img2).squeeze()

            # Convert outputs to binary predictions
            predictions = (outputs > 0.5).float()

            # Ensure labels and predictions have the same shape
            labels = labels.squeeze()

            # Update the correct count
            correct += (predictions == labels).sum().item()

            # Update the total count
            total += labels.size(0)

    # Compute accuracy
    accuracy = correct / total
    print(f"[Prediction]: Accuracy: {accuracy:.4f}")
    return accuracy
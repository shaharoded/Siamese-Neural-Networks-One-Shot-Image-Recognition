import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import SiameseNetworkDataset, get_dataloader
from model import SiameseNetwork, train, predict
from utils import imshow
from config import *

def visualize_data():
    """Function to visualize a batch of data from the Siamese Network Dataset."""
    # Create the DataLoader using the configuration
        # Create the dataset
    dataset = SiameseNetworkDataset(root_dir=DATA_ROOT, file_list=TRAIN_FILE, transform=None, image_size=IMAGE_SIZE)

    # Create a DataLoader from the dataset
    dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Get a batch of data
    data_iter = iter(dataloader)
    example_batch = next(data_iter)

    # Concatenate images for visualization
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    
    # Generate labels for the batch
    labels = ["Twin" if label.item() == 1 else "Not Twin" for label in example_batch[2]]
    
    # Visualize the grid with labels
    imshow(torchvision.utils.make_grid(concatenated), labels=labels)
    

def train_model():
        # Initialize the dataset
    print("[Setup]: Initializing dataset...")
    dataset = SiameseNetworkDataset(
        root_dir=DATA_ROOT,
        file_list=TRAIN_FILE,
        image_size=IMAGE_SIZE
    )

    # Initialize the model
    print("[Setup]: Initializing model...")
    model = SiameseNetwork(CNN_BLOCKS, FC_LAYERS).to(DEVICE)

    # Start training
    print("[Training]: Starting training process...")
    train_losses, val_losses = train(
        model=model,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        epochs=(MIN_EPOCHS, MAX_EPOCHS),
        lr=LEARNING_RATE,
        l2_reg=L2_REG,
        early_stop_patience=EARLY_STOP_PATIENCE,
        save_path=SAVE_PATH,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        qa_mode=True
    )

    print("[Training]: Training complete. Model saved.")
    
        # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    

def main_predict():
    """
    Main function to handle predictions using a trained Siamese Neural Network.
    Function will print a batch's prediction for inspection
    """
    # Initialize the test dataset
    print("[Setup]: Initializing test dataset...")
    test_dataset = SiameseNetworkDataset(
        root_dir=DATA_ROOT,
        file_list=TEST_FILE,
        image_size=IMAGE_SIZE
    )

    # Run predictions
    print("[Prediction]: Running predictions...")
    _ = predict(
        model_path=SAVE_PATH,
        cnn_blocks=CNN_BLOCKS,
        fc_layers=FC_LAYERS,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE
    )

    # Visualize a few samples from the test set
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    data_iter = iter(test_loader)
    img1, img2, label = next(data_iter)

    # Load the model for prediction
    model = SiameseNetwork(CNN_BLOCKS, FC_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        outputs = model(img1.to(DEVICE), img2.to(DEVICE)).squeeze()
        similarity_scores = [f"Sim: {score:.2f}" for score in outputs.cpu().tolist()]
        concatenated = torch.cat((img1, img2), 0)
        imshow(torchvision.utils.make_grid(concatenated), labels=similarity_scores)

        
def view_mistakes(k=10):
    """
    Main function to handle predictions on the test dataset and display the k most mismatched predictions.
    Mismatched predictions are ranked by the absolute error between the predicted similarity score 
    and the actual label.

    Args:
        k (int): Number of mistakes to display.
    """
    # Initialize the test dataset
    print("[Setup]: Initializing test dataset...")
    test_dataset = SiameseNetworkDataset(
        root_dir=DATA_ROOT,
        file_list=TEST_FILE,
        image_size=IMAGE_SIZE
    )

    # Load the test DataLoader
    test_loader = get_dataloader(test_dataset, batch_size=1, num_workers=NUM_WORKERS)

    # Load the trained model
    print("[Prediction]: Loading model...")
    model = SiameseNetwork(CNN_BLOCKS, FC_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Collect predictions and labels
    mismatches = []  # Store tuples: (abs_error, img1, img2, score, label)
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)

            # Predict similarity score
            outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
            abs_error = torch.abs(outputs - labels).item()

            # Store the mismatch details
            mismatches.append((abs_error, img1.cpu(), img2.cpu(), outputs.item(), labels.item()))

    # Sort mismatches by absolute error in descending order
    mismatches.sort(key=lambda x: x[0], reverse=True)

    # Display the top k mismatches
    print(f"\n[Output]: Displaying top {k} mismatches:")
    for i, (abs_error, img1, img2, score, label) in enumerate(mismatches[:k]):
        print(f"\nMistake {i+1}:")
        print(f"  Predicted Score: {score:.2f}")
        print(f"  Actual Label: {label:.0f}")
        print(f"  Absolute Error: {abs_error:.2f}")

        # Visualize the mismatched images
        concatenated = torch.cat((img1, img2), 0)
        imshow(torchvision.utils.make_grid(concatenated), labels=[f"Pred: {score:.2f}", f"Label: {label:.0f}"])
    

if __name__ == "__main__":
    
    def menu():
        print("\nMenu:")
        print("1. Visualize Data")
        print("2. Train a Model")
        print("3. Test Existing Model")
        print("4. View main Mistakes")
        print("5. Exit")

    while True:
        menu()
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")
            continue

        if choice == 1:
            visualize_data()
        elif choice == 2:
            train_model()
        elif choice == 3:
            main_predict()
        elif choice == 4:
            view_mistakes(k=10)
        elif choice == 5:
            print("\nExiting the program. Goodbye!")
        else:
            print("Invalid choice. Please select a valid option.")
        sys.exit()
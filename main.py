'''
Fix the resize to input 105X105 to the original architecture
Fix the backprop to apply with the additional layer
'''


import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import SiameseNetworkDataset, get_dataloader
from model import SiameseNetwork, train, predict
from utils import imshow, calculate_cnn_output_size
from config import *

def visualize_data():
    """Function to visualize a batch of data from the Siamese Network Dataset."""
    # Create the DataLoader using the configuration
        # Create the dataset
    dataset = SiameseNetworkDataset(root_dir=DATA_ROOT, file_list=TRAIN_FILE, transform=None, image_size=IMAGE_SIZE)

    # Create a DataLoader from the dataset
    dataloader = get_dataloader(dataset, batch_size=8, num_workers=NUM_WORKERS)

    # Get a batch of data
    data_iter = iter(dataloader)
    example_batch = next(data_iter)
    img1, img2, labels = example_batch

    # Stack each pair of images vertically and then concatenate the pairs horizontally
    pairs = [torch.cat((img1[i], img2[i]), dim=1) for i in range(img1.size(0))]
    concatenated = torch.cat(pairs, dim=2)

    # Generate labels for the batch
    labels = ["Twin" if label.item() == 1 else "Not Twin" for label in labels]

    # Visualize the grid with labels
    imshow(torchvision.utils.make_grid(concatenated), labels=labels)

  
def calculate_conv_out():
    h, w, c = calculate_cnn_output_size(CNN_BLOCKS, IMAGE_SIZE)
    print(f"Output size after Conv + Pooling: {h}X{w}X{c}={h*w*c}")

    
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
        augment_ratio=AUGMENT_RATIO,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        l2_reg=L2_REG,
        early_stop_patience=EARLY_STOP_PATIENCE,
        save_path=SAVE_PATH,
        num_workers=NUM_WORKERS,
        device=DEVICE
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
        batch_size=4, # For pictures visability
        num_workers=NUM_WORKERS,
        device=DEVICE
    )

    # Visualize a few samples from the test set
    test_loader = get_dataloader(test_dataset, batch_size=4, num_workers=NUM_WORKERS)
    data_iter = iter(test_loader)
    img1, img2, labels = next(data_iter)
    actuals = ['Same' if label.item() == 1 else 'Different' for label in labels]

    # Load the model for prediction
    model = SiameseNetwork(CNN_BLOCKS, FC_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        outputs = model(img1.to(DEVICE), img2.to(DEVICE)).squeeze().cpu().tolist()  # Convert to list
        similarity_scores = [f"{actual}, Pred: {score:.2f}" for actual, score in zip(actuals, outputs)]

        # Stack each pair of images vertically
        pairs = [torch.cat((img1[i], img2[i]), dim=1) for i in range(img1.size(0))]  # dim=1 for vertical stacking
        concatenated = torch.cat(pairs, dim=2)  # Concatenate horizontally

        # Display the grid of stacked pairs with labels
        imshow(concatenated, labels=similarity_scores)

        
def view_mistakes(k=5):
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
    all_scores = []  # Collect all predicted scores
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)

            # Predict similarity score
            outputs, labels = model(img1, img2).squeeze(), labels.squeeze() # Remove extra dimension
            abs_error = torch.abs(outputs - labels).item()

            # Store the mismatch details
            mismatches.append((abs_error, img1.cpu(), img2.cpu(), outputs.item(), labels.item()))

            # Collect scores and labels for analysis
            all_scores.append(outputs.item())
            
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
    
    # Plot the distribution of predictions
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=20, alpha=0.7, label="Predicted Scores", color='blue')
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Scores")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)  # Explicitly set the x-axis range to [0, 1]
    plt.show()

    print("[Analysis]: Prediction confidence distribution plotted.")

if __name__ == "__main__":
    
    def menu():
        print("\nMenu:")
        print("1. Visualize Data")
        print("2. Calculate Convolution Output")
        print("3. Train a Model")
        print("4. Test Existing Model")
        print("5. View Main Mistakes")
        print("9. Exit")

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
            calculate_conv_out()
        elif choice == 3:
            train_model()
        elif choice == 4:
            main_predict()
        elif choice == 5:
            view_mistakes()
        elif choice == 9:
            print("\nExiting the program. Goodbye!")
        else:
            print("Invalid choice. Please select a valid option.")
        sys.exit()
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None, image_size=(105,105)):
        """
        Dataset for Siamese Network.
        Handles same-folder (twins) and different-folder (not twins) pairs.
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(image_size),
        transforms.ToTensor()
        ])        
        self.data = self.__load_data(file_list)


    def __load_data(self, file_list):
        """
        Load data from the file_list (train.txt/test.txt).
        Detects whether pairs are same-folder or different-folder.
        """
        data = []
        total_rows = 0
        positive_count = 0
        negative_count = 0

        with open(file_list, "r") as f:
            for i, line in enumerate(f.readlines()[1:]):  # Skip the header
                total_rows += 1
                parts = line.strip().split("\t")

                # Same-folder pairs (e.g., Abdullah_Gul 13 14)
                if len(parts) == 3:
                    folder, img1, img2 = parts
                    try:
                        img1, img2 = int(img1), int(img2)
                        data.append((folder, img1, img2, 1))  # Same person (label = 1)
                        positive_count += 1
                    except ValueError:
                        print(f"[Dataloader Status]: Skipping malformed line {i+1}: {line.strip()}")

                # Different-folder pairs (e.g., Seth_Gorney 1 Wilton_Gregory 1)
                elif len(parts) == 4:
                    person1, img1, person2, img2 = parts
                    try:
                        img1, img2 = int(img1), int(img2)
                        data.append(((person1, person2), img1, img2, 0))  # Different people (label = 0)
                        negative_count += 1
                    except ValueError:
                        print(f"[Dataloader Status]: Skipping malformed line {i+1}: {line.strip()}")

                else:
                    print(f"[Dataloader Status]: Skipping malformed line {i+1}: {line.strip()}")

        print(f"[Dataloader Status]: Total number of pairs (rows in file): {total_rows}")
        print(f"[Dataloader Status]: Total positives detected: {positive_count}")
        print(f"[Dataloader Status]: Total negatives detected: {negative_count}")

        return data


    def __getitem__(self, index):
        entry = self.data[index]

        # Same-folder pairs
        if isinstance(entry[0], str):
            folder, img1_id, img2_id, same_person = entry
            img1_path = os.path.join(self.root_dir, folder, f"{folder}_{img1_id:04d}.jpg")
            img2_path = os.path.join(self.root_dir, folder, f"{folder}_{img2_id:04d}.jpg")

        # Different-folder pairs
        else:
            (person1, person2), img1_id, img2_id, same_person = entry
            img1_path = os.path.join(self.root_dir, person1, f"{person1}_{img1_id:04d}.jpg")
            img2_path = os.path.join(self.root_dir, person2, f"{person2}_{img2_id:04d}.jpg")

        # Open images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Convert the label (same person: 1, different person: 0) to a tensor
        label = torch.tensor([same_person], dtype=torch.float32)
        return img1, img2, label

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, batch_size, num_workers):
    """
    Create a DataLoader from an existing dataset.
    Shuffles the dataset, as it's sorted by label.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


def stratified_split(dataset, val_split):
    """
    Perform a stratified split of the dataset into training and validation sets.
    
    Args:
        dataset: The full dataset to split.
        val_split: Fraction of data to use for validation.
        
    Returns:
        train_dataset, val_dataset: Subsets for training and validation.
    """
    # Separate indices by label
    label_1_indices = [i for i, data in enumerate(dataset) if data[2].item() == 1]  # Label = 1
    label_0_indices = [i for i, data in enumerate(dataset) if data[2].item() == 0]  # Label = 0

    # Shuffle indices
    np.random.shuffle(label_1_indices)
    np.random.shuffle(label_0_indices)

    # Calculate split sizes
    val_size_1 = int(len(label_1_indices) * val_split)
    val_size_0 = int(len(label_0_indices) * val_split)

    # Split indices
    val_indices = label_1_indices[:val_size_1] + label_0_indices[:val_size_0]
    train_indices = label_1_indices[val_size_1:] + label_0_indices[val_size_0:]

    # Shuffle final indices for train and validation
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def count_labels(dataset):
    '''
    Function for QA on the split operation
    '''
    positives = sum(1 for _, _, label in dataset if label.item() == 1)
    negatives = len(dataset) - positives
    return positives, negatives
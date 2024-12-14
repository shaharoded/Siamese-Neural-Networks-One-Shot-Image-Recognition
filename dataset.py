import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from utils import apply_augmentation

class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, file_list=None, transform=None, image_size=(105,105)):
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
        self.data = self.__load_data(file_list) if file_list else []


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

        print(f"[Dataloader Info]: Total number of pairs (rows in file): {total_rows}")
        print(f"[Dataloader Info]: Total positives detected: {positive_count}")
        print(f"[Dataloader Info]: Total negatives detected: {negative_count}")

        return data
    
    
    def _load_images(self, folder_or_persons, img1_id, img2_id):
        """
        Load the original images from the dataset based on metadata.

        Args:
            folder_or_persons: Folder name or tuple of person names for the images.
            img1_id (int): ID of the first image.
            img2_id (int): ID of the second image.

        Returns:
            Tuple[Image, Image]: The two loaded images as PIL.Image objects.
        """
        # Same-folder pairs
        if isinstance(folder_or_persons, str):
            folder = folder_or_persons
            img1_path = os.path.join(self.root_dir, folder, f"{folder}_{img1_id:04d}.jpg")
            img2_path = os.path.join(self.root_dir, folder, f"{folder}_{img2_id:04d}.jpg")
        # Different-folder pairs
        else:
            person1, person2 = folder_or_persons
            img1_path = os.path.join(self.root_dir, person1, f"{person1}_{img1_id:04d}.jpg")
            img2_path = os.path.join(self.root_dir, person2, f"{person2}_{img2_id:04d}.jpg")

        # Load images as PIL objects
        img1 = Image.open(img1_path).convert("L")  # Convert to grayscale
        img2 = Image.open(img2_path).convert("L")  # Convert to grayscale
        return img1, img2
    
    
    def __getitem__(self, index):
        '''
        Handles both in-memory (augmented pairs) and actual pairs
        '''
        entry = self.data[index]

        # Check if the entry contains in-memory augmented images
        if isinstance(entry[1], Image.Image) and isinstance(entry[2], Image.Image):
            # Augmented pair: entry = (folder_or_persons, img1_aug, img2_aug, label)
            img1, img2, same_person = entry[1], entry[2], entry[3]

            # Apply transformations if defined
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # Convert label to tensor
            label = torch.tensor([same_person], dtype=torch.float32)
            return img1, img2, label

        # Original pair: Process using paths
        else:
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
            img1 = Image.open(img1_path).convert("L")
            img2 = Image.open(img2_path).convert("L")

            # Apply transformations if defined
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # Convert label to tensor
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
    Perform a stratified split of the dataset into independent training and validation datasets.
    
    Args:
        dataset (SiameseNetworkDataset): The full dataset to split.
        val_split (float): Fraction of data to use for validation.
        
    Returns:
        train_dataset (SiameseNetworkDataset): Training dataset.
        val_dataset (SiameseNetworkDataset): Validation dataset.
    """
    # Separate data by label
    label_1_data = [data for data in dataset.data if data[3] == 1]  # Label = 1
    label_0_data = [data for data in dataset.data if data[3] == 0]  # Label = 0

    # Shuffle data
    np.random.shuffle(label_1_data)
    np.random.shuffle(label_0_data)

    # Calculate split sizes
    val_size_1 = int(len(label_1_data) * val_split)
    val_size_0 = int(len(label_0_data) * val_split)

    # Split data
    val_data = label_1_data[:val_size_1] + label_0_data[:val_size_0]
    train_data = label_1_data[val_size_1:] + label_0_data[val_size_0:]

    # Shuffle final train and validation data
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Create new dataset objects
    train_dataset = SiameseNetworkDataset(
        root_dir=dataset.root_dir,
        file_list=None,  # No file list needed as we manually split
        transform=dataset.transform,
        image_size=dataset.transform.transforms[1].size if dataset.transform else None,
    )
    val_dataset = SiameseNetworkDataset(
        root_dir=dataset.root_dir,
        file_list=None,  # No file list needed as we manually split
        transform=dataset.transform,
        image_size=dataset.transform.transforms[1].size if dataset.transform else None,
    )

    # Assign the split data directly
    train_dataset.data = train_data
    val_dataset.data = val_data

    return train_dataset, val_dataset


def augment_dataset(dataset, augment_ratio):
    """
    Augments the dataset by creating additional pairs with augmented images.
    The augmented images will be generated dynamically.

    Args:
        dataset (SiameseNetworkDataset): The dataset to augment.
        augment_ratio (float): Ratio of augmented pairs to add for each original pair.

    Returns:
        list: Augmented metadata containing both original and augmented pairs.
    """
    print(f"[Augmentation]: Augmenting dataset with ratio {augment_ratio}...")

    original_data = dataset.data.copy()  # Start with the original data
    augmented_data = original_data.copy()

    for folder_or_persons, img1_id, img2_id, label in original_data:
        for _ in range(int(augment_ratio - 1)):
            # Dynamically load the original images
            img1, img2 = dataset._load_images(folder_or_persons, img1_id, img2_id)

            # Apply augmentations to both images
            img1_aug = apply_augmentation(img1)
            img2_aug = apply_augmentation(img2)

            # Save augmented images as temporary PIL images
            augmented_data.append((folder_or_persons, img1_aug, img2_aug, label))

    print(f"[Augmentation]: Dataset augmented. Original size: {len(original_data)}, New size: {len(augmented_data)}")
    return augmented_data


def count_labels(dataset):
    '''
    Function for QA on the split operation
    '''
    positives = sum(1 for _, _, label in dataset if label.item() == 1)
    negatives = len(dataset) - positives
    return positives, negatives
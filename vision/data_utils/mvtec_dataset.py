import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MVTecADDataset(Dataset):
    def __init__(self, root_dir, category, transform=None, target_transform=None, is_train=True, resize=256):
        """
        Args:
            root_dir (string): Directory with all the MVTec AD data.
            category (string): The category of the dataset (e.g., 'bottle', 'cable').
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on a mask.
            is_train (bool): If True, loads training data. If False, loads test data.
            resize (int): Desired size for images.
        """
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize

        self.image_paths = []
        self.ground_truth_paths = []
        self.labels = [] # 0 for good, 1 for anomaly

        self.default_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.default_target_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=Image.NEAREST), # Use NEAREST for masks
            transforms.ToTensor(),
        ])

        self._load_dataset()
        
        logging.info(f"Loaded {len(self.image_paths)} images for MVTec AD category '{category}' (Train: {is_train})")

    def _load_dataset(self):
        category_dir = os.path.join(self.root_dir, self.category)
        if not os.path.exists(category_dir):
            logging.error(f"Category directory not found: {category_dir}")
            raise FileNotFoundError(f"MVTec AD category '{self.category}' not found at '{category_dir}'")

        if self.is_train:
            # Load training images (only 'good' samples)
            train_good_dir = os.path.join(category_dir, 'train', 'good')
            if not os.path.exists(train_good_dir):
                logging.error(f"Train good directory not found: {train_good_dir}")
                raise FileNotFoundError(f"Train good directory not found for category '{self.category}'")
            
            self.image_paths = sorted(glob.glob(os.path.join(train_good_dir, '*.png')))
            self.labels = [0] * len(self.image_paths) # All training samples are 'good'
            self.ground_truth_paths = [None] * len(self.image_paths) # No GT masks for training
        else:
            # Load test images (good and anomalous) and their ground truth masks
            test_base_dir = os.path.join(category_dir, 'test')
            gt_base_dir = os.path.join(category_dir, 'ground_truth')

            if not os.path.exists(test_base_dir):
                logging.error(f"Test base directory not found: {test_base_dir}")
                raise FileNotFoundError(f"Test base directory not found for category '{self.category}'")
            # ground_truth dir is optional, only present for anomalous types
            
            # Load 'good' test images
            good_test_dir = os.path.join(test_base_dir, 'good')
            if os.path.exists(good_test_dir):
                for img_path in sorted(glob.glob(os.path.join(good_test_dir, '*.png'))):
                    self.image_paths.append(img_path)
                    self.ground_truth_paths.append(None) # Good images have no GT mask
                    self.labels.append(0)
            else:
                logging.warning(f"Good test directory not found: {good_test_dir}. Skipping good test images.")

            # Load anomalous test images and their ground truth masks
            # Exclude 'good' and 'ground_truth' subdirectories from anomaly types
            subdirs = [d for d in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, d))]
            anomaly_types = [d for d in subdirs if d not in ['good', 'ground_truth']]

            for anomaly_type in anomaly_types:
                anomaly_dir = os.path.join(test_base_dir, anomaly_type)
                
                # MVTec stores images and masks with corresponding names (e.g., 000.png and 000_mask.png)
                # We need to list images and find their corresponding masks
                anomaly_image_files = sorted(os.listdir(anomaly_dir))

                for img_file in anomaly_image_files:
                    if not img_file.endswith('.png'):
                        continue # Skip non-image files

                    img_path = os.path.join(anomaly_dir, img_file)
                    
                    # Construct ground truth mask path
                    gt_file = img_file.replace('.png', '_mask.png')
                    current_gt_path = os.path.join(gt_base_dir, anomaly_type, gt_file)

                    if not os.path.exists(current_gt_path):
                        logging.warning(f"Ground truth mask not found for {img_path} at {current_gt_path}. Skipping.")
                        continue # Skip if GT mask is missing

                    self.image_paths.append(img_path)
                    self.ground_truth_paths.append(current_gt_path)
                    self.labels.append(1) # Label as anomalous

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply image transform
        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)

        # Load and apply mask transform if available
        # Masks are single channel (L) and need to be binarized (0 or 1)
        mask = torch.zeros((1, self.resize, self.resize), dtype=torch.float32) # Default empty mask
        gt_path = self.ground_truth_paths[idx]
        if gt_path is not None and os.path.exists(gt_path):
            gt_mask_image = Image.open(gt_path).convert('L') # Convert to grayscale
            if self.target_transform:
                mask = self.target_transform(gt_mask_image)
            else:
                mask = self.default_target_transform(gt_mask_image)
            mask = (mask > 0).float() # Binarize mask: 0 for background, 1 for anomaly

        return image, mask, label

if __name__ == "__main__":
    # Example usage:
    # This assumes the 'bottle' dataset is downloaded and extracted into 'data/mvtec_ad/bottle'
    # The structure should be:
    # data/mvtec_ad/bottle/train/good/...
    # data/mvtec_ad/bottle/test/good/...
    # data/mvtec_ad/bottle/test/<anomaly_type>/... (e.g., broken_mouth)
    # data/mvtec_ad/bottle/ground_truth/<anomaly_type>/... (e.g., broken_mouth)

    root_dir = "data/mvtec_ad"
    category = "bottle"
    resize_size = 256

    try:
        # --- Training Dataset ---
        print("\n--- Initializing Training Dataset ---")
        train_dataset = MVTecADDataset(root_dir=root_dir, category=category, is_train=True, resize=resize_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

        print("\n--- Training Data Sample (first batch) ---")
        for i, (images, masks, labels) in enumerate(train_loader):
            print(f"Batch {i}: images shape={images.shape}, masks shape={masks.shape}, labels={labels}")
            if i == 0:
                break
        
        # --- Test Dataset ---
        print("\n--- Initializing Test Dataset ---")
        test_dataset = MVTecADDataset(root_dir=root_dir, category=category, is_train=False, resize=resize_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

        print("\n--- Test Data Sample (first few batches) ---")
        for i, (images, masks, labels) in enumerate(test_loader):
            print(f"Batch {i}: images shape={images.shape}, masks shape={masks.shape}, labels={labels}")
            if any(labels.numpy() == 1):
                print(f"  -> Found anomalous sample(s) in batch, mask sum: {masks.sum()}")
            if i >= 4: # Print first 5 test batches
                break

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the MVTec AD 'bottle' dataset is correctly downloaded and extracted.")
        print(f"Expected structure: {root_dir}/{category}/train/good/..., {root_dir}/{category}/test/good/..., etc.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

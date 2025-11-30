# Load and aggregate multiple medical imaging datasets

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

from codebase.data.augmentation import get_train_transforms, get_val_test_transforms


# Constants
REQUIRED_COLUMNS = ['filepath', 'label']
LABEL_MAPPING = {
    'benign': 0,
    'malignant': 1,
    'Benign': 0,
    'Malignant': 1,
    'BENIGN': 0,
    'MALIGNANT': 1
}
VALID_LABELS = {0, 1}  # Binary classification: 0=benign, 1=malignant


class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset for skin lesion images loaded from CSV or DataFrame.
    
    This dataset handles:
    - Loading images from file paths
    - Converting text labels to numeric (benign/malignant -> 0/1)
    - Applying image transforms (augmentation for training, preprocessing for val/test)
    - Error handling for missing or corrupted images
    
    Example:
        >>> dataset = SkinLesionDataset(csv_path='data/labels.csv', transform=my_transform)
        >>> image, label = dataset[0]  # Get first sample
    """
    
    def __init__(
        self, 
        csv_path: Optional[str] = None, 
        df: Optional[pd.DataFrame] = None, 
        image_dir: Optional[str] = None, 
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with 'filepath' and 'label' columns 
                     (required if df not provided)
            df: DataFrame with 'filepath' and 'label' columns 
                (required if csv_path not provided)
            image_dir: Optional base directory to prepend to filepaths
                      (useful if CSV has relative paths)
            transform: Optional transform pipeline to apply to images
                     (typically from augmentation.get_train_transforms() or 
                      augmentation.get_val_test_transforms())
        
        Raises:
            ValueError: If neither csv_path nor df is provided, or if required 
                       columns are missing, or if labels are invalid
            FileNotFoundError: If csv_path is provided but file doesn't exist
        """
        # Step 1: Load data from CSV or DataFrame
        self.df = self._load_data(csv_path, df)
        
        # Step 2: Validate that required columns exist
        self._validate_columns()
        
        # Step 3: Store configuration
        self.image_dir = image_dir
        self.transform = transform
        
        # Step 4: Encode labels (convert text to numeric if needed)
        self._encode_labels()
        
        # Step 5: Validate labels are correct (0 or 1)
        self._validate_labels()
    
    def _load_data(self, csv_path: Optional[str], df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV file or DataFrame."""
        if df is not None:
            return df.copy()
        elif csv_path is not None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}\n"
                    f"Please ensure the file exists and the path is correct."
                )
            return pd.read_csv(csv_path)
        else:
            raise ValueError(
                "Either 'csv_path' or 'df' must be provided.\n"
                "Example: SkinLesionDataset(csv_path='data/labels.csv')"
            )
    
    def _validate_columns(self) -> None:
        """Check that required columns exist in the DataFrame."""
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_columns:
            available_cols = ', '.join(self.df.columns.tolist())
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Required columns: {REQUIRED_COLUMNS}\n"
                f"Available columns: {available_cols}"
            )
    
    def _encode_labels(self) -> None:
        """
        Convert text labels to numeric if needed.
        
        Handles labels like: 'benign', 'Benign', 'BENIGN' -> 0
                            'malignant', 'Malignant', 'MALIGNANT' -> 1
        """
        # Check if labels are already numeric
        if pd.api.types.is_numeric_dtype(self.df['label']):
            # Already numeric, convert to int
            self.df['label'] = self.df['label'].astype(int)
        else:
            # Text labels - convert to numeric using mapping
            self.df['label'] = self.df['label'].map(LABEL_MAPPING)
            
            # Check for unmapped labels (None values indicate unknown labels)
            unmapped_mask = self.df['label'].isna()
            if unmapped_mask.any():
                unmapped_values = self.df.loc[unmapped_mask, 'label'].unique().tolist()
                raise ValueError(
                    f"Unknown label values found: {unmapped_values}\n"
                    f"Expected values: {list(LABEL_MAPPING.keys())} or numeric (0, 1)\n"
                    f"Please check your CSV file and update the labels."
                )
    
    def _validate_labels(self) -> None:
        """Ensure all labels are valid (0 or 1 for binary classification)."""
        invalid_labels = set(self.df['label'].unique()) - VALID_LABELS
        if invalid_labels:
            raise ValueError(
                f"Invalid label values found: {invalid_labels}\n"
                f"Expected values: {VALID_LABELS} (0=benign, 1=malignant)"
            )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample (0 to len(dataset)-1)
            
        Returns:
            tuple: (image_tensor, label) where:
                - image_tensor: torch.Tensor of shape (C, H, W) - transformed image
                - label: int (0 for benign, 1 for malignant)
        
        Raises:
            FileNotFoundError: If image file cannot be found or opened
            IndexError: If idx is out of range
        """
        # Validate index
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.df)} samples.")
        
        # Get filepath and label from DataFrame
        filepath = self.df.iloc[idx]['filepath']
        label = int(self.df.iloc[idx]['label'])
        
        # Construct full path to image
        full_path = self._get_image_path(filepath)
        
        # Load and convert image
        image = self._load_image(full_path)
        
        # Apply transforms (augmentation for training, preprocessing for val/test)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _get_image_path(self, filepath: str) -> str:
        """Construct full path to image file."""
        if self.image_dir:
            # If image_dir is provided, join it with the filepath
            return os.path.join(self.image_dir, filepath)
        else:
            # Otherwise, use filepath as-is (can be absolute or relative)
            return filepath
    
    def _load_image(self, full_path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            full_path: Full path to image file
            
        Returns:
            PIL.Image: RGB image
            
        Raises:
            FileNotFoundError: If image file doesn't exist or cannot be opened
        """
        try:
            image = Image.open(full_path)
            # Convert to RGB (handles grayscale, RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {full_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading image at {full_path}: {type(e).__name__}: {e}\n"
                f"Please check that the file exists and is a valid image format."
            )


def split_dataset(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification.
    
    This ensures that each split maintains the same class distribution as the 
    original dataset (important for imbalanced datasets).
    
    Args:
        df: DataFrame with 'filepath' and 'label' columns
        train_ratio: Proportion for training set (default: 0.7 = 70%)
        val_ratio: Proportion for validation set (default: 0.15 = 15%)
        test_ratio: Proportion for test set (default: 0.15 = 15%)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df) - Three DataFrames with same columns as input
        
    Raises:
        ValueError: If ratios don't sum to 1.0, or if dataset is too small to split
    """
    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, but got {total_ratio:.6f}\n"
            f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
    
    # Check dataset size
    if len(df) < 3:
        raise ValueError(
            f"Dataset too small to split (only {len(df)} samples). "
            f"Need at least 3 samples for train/val/test split."
        )
    
    # Check if we have enough samples for stratification
    min_class_count = df['label'].value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            f"Cannot perform stratified split: one class has only {min_class_count} sample(s). "
            f"Need at least 2 samples per class."
        )
    
    # Step 1: Split train from (val + test)
    # This gives us 70% train, 30% (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),  # 0.15 + 0.15 = 0.30
        stratify=df['label'],  # Maintain class balance
        random_state=random_state
    )
    
    # Step 2: Split (val + test) into val and test
    # Within the 30%, we want 15% val and 15% test
    # So val_size = 0.15 / 0.30 = 0.5 (50% of the temp_df)
    val_size = val_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),  # 1 - 0.5 = 0.5 (50% of temp_df goes to test)
        stratify=temp_df['label'],  # Maintain class balance
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def get_dataloaders(
    config, 
    csv_path: str = 'data/labels.csv', 
    image_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    This is the main function you'll call from your training script. It:
    1. Loads the CSV file
    2. Splits data into train/val/test (70/15/15)
    3. Creates datasets with appropriate transforms
    4. Wraps them in DataLoaders with correct settings
    
    Args:
        config: TrainingConfig object with batch_size attribute
        csv_path: Path to CSV file with 'filepath' and 'label' columns
                 (default: 'data/labels.csv')
        image_dir: Optional base directory for images 
                  (useful if CSV has relative paths)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - Three PyTorch DataLoaders
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is empty or has invalid data
    
    Example:
        >>> from codebase.config import TrainingConfig
        >>> config = TrainingConfig(batch_size=32)
        >>> train_loader, val_loader, test_loader = get_dataloaders(config)
    """
    # Step 1: Validate CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at: {csv_path}\n"
            f"Please create a CSV file with 'filepath' and 'label' columns.\n"
            f"Example CSV format:\n"
            f"  filepath,label\n"
            f"  data/images/img_001.jpg,0\n"
            f"  data/images/img_002.jpg,1"
        )
    
    # Step 2: Load and validate dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {csv_path}")
    print(f"{'='*60}")
    
    try:
        full_df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    if len(full_df) == 0:
        raise ValueError(f"CSV file {csv_path} is empty. Please add data.")
    
    # Display dataset info
    print(f"Total samples: {len(full_df)}")
    print(f"\nClass distribution:")
    class_counts = full_df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        percentage = (count / len(full_df)) * 100
        label_name = "benign" if label == 0 or str(label).lower() in ['0', 'benign'] else "malignant"
        print(f"  {label_name} ({label}): {count} samples ({percentage:.1f}%)")
    
    # Step 3: Split into train/val/test (70/15/15)
    print(f"\n{'='*60}")
    print("Splitting dataset into train/validation/test...")
    print(f"{'='*60}")
    
    try:
        train_df, val_df, test_df = split_dataset(
            full_df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
    except Exception as e:
        raise ValueError(f"Error splitting dataset: {e}")
    
    print(f"Split sizes:")
    print(f"  Train:      {len(train_df):5d} samples ({len(train_df)/len(full_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):5d} samples ({len(val_df)/len(full_df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):5d} samples ({len(test_df)/len(full_df)*100:.1f}%)")
    
    # Step 4: Create datasets with appropriate transforms
    print(f"\n{'='*60}")
    print("Creating datasets...")
    print(f"{'='*60}")
    
    train_dataset = SkinLesionDataset(
        df=train_df,
        image_dir=image_dir,
        transform=get_train_transforms()  # With augmentation
    )
    print(f"✓ Train dataset created (with augmentation)")
    
    val_dataset = SkinLesionDataset(
        df=val_df,
        image_dir=image_dir,
        transform=get_val_test_transforms()  # No augmentation
    )
    print(f"✓ Validation dataset created (no augmentation)")
    
    test_dataset = SkinLesionDataset(
        df=test_df,
        image_dir=image_dir,
        transform=get_val_test_transforms()  # No augmentation
    )
    print(f"✓ Test dataset created (no augmentation)")
    
    # Step 5: Create DataLoaders
    print(f"\n{'='*60}")
    print("Creating DataLoaders...")
    print(f"{'='*60}")
    
    # Configure DataLoader settings
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()  # Faster GPU transfer if available
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader
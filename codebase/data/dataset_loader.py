# Load and aggregate multiple medical imaging datasets

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

from codebase.utils.helpers import get_train_transforms, get_val_test_transforms


# Constants
REQUIRED_COLUMNS = ['filepath', 'label']
LABEL_MAPPING = {
    'benign': 0, 'malignant': 1,
    'Benign': 0, 'Malignant': 1,
    'BENIGN': 0, 'MALIGNANT': 1
}
VALID_LABELS = {0, 1}  # 0=benign, 1=malignant


class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset for skin lesion images from CSV/DataFrame.
    Handles: loading images, label encoding (text->numeric), applying transforms.
    """
    
    def __init__(
        self, 
        csv_path: Optional[str] = None, 
        df: Optional[pd.DataFrame] = None, 
        image_dir: Optional[str] = None, 
        transform: Optional[callable] = None
    ):
        """Initialize dataset from CSV or DataFrame."""
        self.df = self._load_data(csv_path, df)
        self._validate_columns()
        self.image_dir = image_dir
        self.transform = transform
        self._encode_labels()
        self._validate_labels()
    
    def _load_data(self, csv_path: Optional[str], df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV or DataFrame."""
        if df is not None:
            return df.copy()
        elif csv_path is not None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            return pd.read_csv(csv_path)
        else:
            raise ValueError("Either 'csv_path' or 'df' must be provided")
    
    def _validate_columns(self) -> None:
        """Check required columns exist."""
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED_COLUMNS}")
    
    def _encode_labels(self) -> None:
        """Convert text labels to numeric (benign/malignant -> 0/1)."""
        if pd.api.types.is_numeric_dtype(self.df['label']):
            self.df['label'] = self.df['label'].astype(int)
        else:
            self.df['label'] = self.df['label'].map(LABEL_MAPPING)
            if self.df['label'].isna().any():
                unmapped = self.df.loc[self.df['label'].isna(), 'label'].unique()
                raise ValueError(f"Unknown labels: {unmapped}. Expected: {list(LABEL_MAPPING.keys())} or 0/1")
    
    def _validate_labels(self) -> None:
        """Ensure labels are 0 or 1."""
        invalid = set(self.df['label'].unique()) - VALID_LABELS
        if invalid:
            raise ValueError(f"Invalid labels: {invalid}. Expected: {VALID_LABELS}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample: (image_tensor, label)."""
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range [0, {len(self.df)})")
        
        filepath = self.df.iloc[idx]['filepath']
        label = int(self.df.iloc[idx]['label'])
        full_path = os.path.join(self.image_dir, filepath) if self.image_dir else filepath
        
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading {full_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def split_dataset(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test with stratification (maintains class balance).
    
    Returns:
        (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if len(df) < 3:
        raise ValueError(f"Dataset too small: {len(df)} samples. Need at least 3.")
    
    min_class_count = df['label'].value_counts().min()
    if min_class_count < 2:
        raise ValueError(f"Cannot stratify: one class has only {min_class_count} sample(s)")
    
    # Split train from (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df['label'], random_state=random_state
    )
    
    # Split val from test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size),
        stratify=temp_df['label'], random_state=random_state
    )
    
    return train_df, val_df, test_df


def get_dataloaders(
    config, 
    csv_path: str = 'codebase/data/labels.csv', 
    image_dir: Optional[str] = 'codebase/data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        config: TrainingConfig with batch_size
        csv_path: Path to CSV with 'filepath' and 'label' columns
        image_dir: Optional base directory for images
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {csv_path}")
    print(f"{'='*60}")
    
    try:
        full_df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")
    
    if len(full_df) == 0:
        raise ValueError(f"CSV is empty: {csv_path}")
    
    print(f"Total samples: {len(full_df)}")
    class_counts = full_df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        pct = (count / len(full_df)) * 100
        name = "benign" if str(label).lower() in ['0', 'benign'] else "malignant"
        print(f"  {name} ({label}): {count} ({pct:.1f}%)")
    
    # Split into train/val/test (70/15/15)
    print(f"\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(full_df, random_state=42)
    print(f"  Train: {len(train_df)} ({len(train_df)/len(full_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(full_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(full_df)*100:.1f}%)")
    
    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = SkinLesionDataset(df=train_df, image_dir=image_dir, transform=get_train_transforms())
    val_dataset = SkinLesionDataset(df=val_df, image_dir=image_dir, transform=get_val_test_transforms())
    test_dataset = SkinLesionDataset(df=test_df, image_dir=image_dir, transform=get_val_test_transforms())
    print(f"✓ Datasets created")
    
    # Create DataLoaders
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    print(f"✓ DataLoaders created (batch_size={config.batch_size}, workers={num_workers})")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader
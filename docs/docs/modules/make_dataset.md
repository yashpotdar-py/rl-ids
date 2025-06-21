# Data Processing Module

## Overview

The data processing module (`rl_ids.make_dataset`) provides comprehensive data preprocessing capabilities for the CICIDS2017 intrusion detection dataset. It handles data loading, cleaning, normalization, balancing, and train/validation/test splitting with advanced techniques for handling class imbalance and outliers.

## Key Features

- **Multi-file CSV Loading** - Automatic detection and combination of multiple CSV files
- **Advanced Data Cleaning** - Handles missing values, infinite values, and outliers  
- **Multiple Normalization Methods** - Standard, Robust, and MinMax scaling options
- **Class Balancing** - SMOTE, SMOTETomek, and undersampling techniques
- **Stratified Splitting** - Maintains class distribution across splits
- **Memory Optimization** - Efficient data type conversion and memory management
- **Progress Tracking** - Visual progress bars for long-running operations

## Classes

### `DataGenerator`

Handles loading and initial preprocessing of raw CSV data files.

**Methods**

#### `load_and_preprocess_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame`

Loads and preprocesses CSV data files from the specified directory.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data_dir` | `Path` | Directory containing CSV files (default: `RAW_DATA_DIR`) |

**Returns**
- `pd.DataFrame`: Combined and preprocessed dataset

**Processing Steps**
1. Loads all CSV files from the directory
2. Combines multiple files into single dataset
3. Removes columns with all null values
4. Fills remaining null values with 0
5. Handles infinite values
6. Encodes categorical labels
7. Optimizes data types for memory efficiency

**Example**
```python
from rl_ids.make_dataset import DataGenerator
from rl_ids.config import RAW_DATA_DIR

# Initialize data generator
generator = DataGenerator()

# Load and preprocess data
data = generator.load_and_preprocess_data(RAW_DATA_DIR)
print(f"Loaded dataset shape: {data.shape}")

# Get label mapping
label_mapping = generator.get_label_mapping()
print(f"Label mapping: {label_mapping}")
```

#### `get_label_mapping() -> dict`

Returns mapping between encoded labels and original label names.

**Returns**
- `dict`: Mapping from encoded integers to original label strings

#### `save_processed_data(output_path: Path, data: pd.DataFrame = None) -> Path`

Saves processed data to specified path.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `output_path` | `Path` | Path to save the processed data |
| `data` | `pd.DataFrame` | Data to save (optional, uses internal data if None) |

**Returns**
- `Path`: Path where data was saved

### `DataProcessor`

Enhanced data processing with multiple normalization and balancing strategies.

**Methods**

#### `normalize_data(df: pd.DataFrame, method: str = 'robust', handle_outliers: bool = True) -> pd.DataFrame`

Normalizes features using specified scaling method with outlier handling.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe to normalize |
| `method` | `str` | Scaling method ('standard', 'robust', 'minmax') |
| `handle_outliers` | `bool` | Whether to handle outliers before normalization |

**Returns**
- `pd.DataFrame`: Normalized dataset

**Normalization Methods**
- **`'standard'`**: StandardScaler - zero mean, unit variance
- **`'robust'`**: RobustScaler - uses median and IQR, robust to outliers  
- **`'minmax'`**: MinMaxScaler - scales to [0,1] range

**Example**
```python
from rl_ids.make_dataset import DataProcessor

processor = DataProcessor()

# Robust normalization (recommended)
normalized_data = processor.normalize_data(
    data, 
    method='robust', 
    handle_outliers=True
)

# Standard normalization
standard_data = processor.normalize_data(
    data, 
    method='standard', 
    handle_outliers=False
)
```

#### `balance_data(df: pd.DataFrame, method: str = 'smote', random_state: int = 42) -> pd.DataFrame`

Balances dataset using specified method to handle class imbalance.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe to balance |
| `method` | `str` | Balancing method ('smote', 'smotetomek', 'undersample') |
| `random_state` | `int` | Random seed for reproducibility |

**Returns**
- `pd.DataFrame`: Balanced dataset

**Balancing Methods**
- **`'smote'`**: SMOTE oversampling - generates synthetic minority samples
- **`'smotetomek'`**: SMOTE + Tomek links - oversampling + cleaning
- **`'undersample'`**: Random undersampling of majority classes

**Example**
```python
# SMOTE balancing (recommended for minority classes)
balanced_data = processor.balance_data(
    normalized_data, 
    method='smote', 
    random_state=42
)

# Combined SMOTE + Tomek cleaning
clean_balanced_data = processor.balance_data(
    normalized_data, 
    method='smotetomek', 
    random_state=42
)
```

#### `split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

Splits data into train/validation/test sets using stratified sampling.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe to split |
| `test_size` | `float` | Proportion for test set (0.0-1.0) |
| `val_size` | `float` | Proportion for validation set (0.0-1.0) |
| `random_state` | `int` | Random seed for reproducibility |

**Returns**
- `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: (train, validation, test) datasets

**Example**
```python
# Split with 60% train, 20% val, 20% test
train_data, val_data, test_data = processor.split_data(
    balanced_data,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

## Command Line Interface

The module provides a CLI for data processing operations:

```bash
# Process data with default settings
python -m rl_ids.make_dataset

# Custom normalization method
python -m rl_ids.make_dataset --normalize-method robust

# Custom balancing method  
python -m rl_ids.make_dataset --balance-method smote

# Custom output directory
python -m rl_ids.make_dataset --output-dir ./custom_output

# Skip balancing step
python -m rl_ids.make_dataset --no-balance

# Enable debug logging
python -m rl_ids.make_dataset --verbose
```

### CLI Options

| Option | Type | Description |
|--------|------|-------------|
| `--input-dir` | `Path` | Input directory with raw CSV files |
| `--output-dir` | `Path` | Output directory for processed files |
| `--normalize-method` | `str` | Normalization method (standard/robust/minmax) |
| `--balance-method` | `str` | Balancing method (smote/smotetomek/undersample) |
| `--test-size` | `float` | Test set proportion |
| `--val-size` | `float` | Validation set proportion |
| `--no-balance` | `bool` | Skip data balancing |
| `--random-state` | `int` | Random seed |
| `--verbose` | `bool` | Enable debug logging |

## Complete Processing Pipeline

### Example: Full Data Processing Workflow

```python
from rl_ids.make_dataset import DataGenerator, DataProcessor
from rl_ids.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path

def process_cicids_data():
    """Complete data processing pipeline"""
    
    # Step 1: Load and preprocess raw data
    generator = DataGenerator()
    raw_data = generator.load_and_preprocess_data(RAW_DATA_DIR)
    
    # Save initial processed data
    processed_file = PROCESSED_DATA_DIR / "cicids2017_processed.csv"
    generator.save_processed_data(processed_file, raw_data)
    
    # Step 2: Advanced processing
    processor = DataProcessor()
    
    # Normalize features
    normalized_data = processor.normalize_data(
        raw_data, 
        method='robust', 
        handle_outliers=True
    )
    
    # Save normalized data
    normalized_file = PROCESSED_DATA_DIR / "cicids2017_normalized.csv"
    normalized_data.to_csv(normalized_file, index=False)
    
    # Balance classes
    balanced_data = processor.balance_data(
        normalized_data, 
        method='smote', 
        random_state=42
    )
    
    # Save balanced data
    balanced_file = PROCESSED_DATA_DIR / "cicids2017_balanced.csv"
    balanced_data.to_csv(balanced_file, index=False)
    
    # Split into train/val/test
    train_data, val_data, test_data = processor.split_data(
        balanced_data,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    
    # Save splits
    train_data.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_data.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    # Get label mapping for reference
    label_mapping = generator.get_label_mapping()
    
    return {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'features': len(processor.feature_columns),
        'classes': len(label_mapping),
        'label_mapping': label_mapping
    }

# Run the pipeline
results = process_cicids_data()
print(f"Processing complete: {results}")
```

## Data Quality and Validation

### Feature Statistics

```python
def analyze_data_quality(df: pd.DataFrame):
    """Analyze data quality metrics"""
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_values = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Infinite values: {inf_values}")
    
    # Class distribution
    if 'Label' in df.columns:
        class_dist = df['Label'].value_counts().sort_index()
        print("Class distribution:")
        for label, count in class_dist.items():
            percentage = count / len(df) * 100
            print(f"  Class {label}: {count:,} ({percentage:.1f}%)")
```

### Data Validation Checks

```python
def validate_processed_data(df: pd.DataFrame) -> bool:
    """Validate processed data meets requirements"""
    
    checks = []
    
    # Check 1: No missing values
    has_missing = df.isnull().sum().sum() > 0
    checks.append(('No missing values', not has_missing))
    
    # Check 2: No infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_infinite = np.isinf(df[numeric_cols]).sum().sum() > 0
    checks.append(('No infinite values', not has_infinite))
    
    # Check 3: Features are numeric
    feature_cols = [col for col in df.columns if col not in ['Label', 'Label_Original']]
    all_numeric = all(df[col].dtype in ['int64', 'float64', 'float32'] for col in feature_cols)
    checks.append(('All features numeric', all_numeric))
    
    # Check 4: Labels are encoded
    if 'Label' in df.columns:
        labels_encoded = df['Label'].dtype in ['int64', 'int32']
        checks.append(('Labels encoded', labels_encoded))
    
    # Print results
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
    
    return all(passed for _, passed in checks)
```

## Performance Optimization

### Memory Efficiency

```python
def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage"""
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col not in ['Label', 'Label_Original']:
            # Convert to float32 if possible
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Check if int32 is sufficient
                if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"Memory optimization: {initial_memory:.1f}MB → {final_memory:.1f}MB")
    print(f"Reduction: {memory_reduction:.1f}%")
    
    return df
```

### Batch Processing

```python
def process_large_dataset_in_chunks(file_path: Path, chunk_size: int = 10000):
    """Process large datasets in chunks to manage memory"""
    
    processor = DataProcessor()
    processed_chunks = []
    
    # Process data in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Apply processing to chunk
        normalized_chunk = processor.normalize_data(chunk, method='robust')
        processed_chunks.append(normalized_chunk)
    
    # Combine processed chunks
    final_data = pd.concat(processed_chunks, ignore_index=True)
    return final_data
```

## Troubleshooting

### Common Issues

1. **Memory Errors with Large Datasets**
   ```python
   # Use chunked processing
   chunk_size = 5000  # Reduce if still getting memory errors
   data = process_large_dataset_in_chunks(data_file, chunk_size)
   ```

2. **Infinite Values in Data**
   ```python
   # Check for infinite values before processing
   inf_cols = df.columns[np.isinf(df.select_dtypes(include=[np.number])).any()]
   print(f"Columns with infinite values: {inf_cols.tolist()}")
   
   # Replace infinite values
   df = df.replace([np.inf, -np.inf], np.nan)
   df = df.fillna(df.median())
   ```

3. **Class Imbalance Issues**
   ```python
   # Check class distribution
   class_counts = df['Label'].value_counts()
   imbalance_ratio = class_counts.max() / class_counts.min()
   
   if imbalance_ratio > 10:
       print(f"High class imbalance detected: {imbalance_ratio:.1f}:1")
       # Use SMOTE or SMOTETomek for balancing
       balanced_data = processor.balance_data(df, method='smotetomek')
   ```

4. **Normalization Failures**
   ```python
   # Check for constant features (zero variance)
   constant_features = df.var() == 0
   if constant_features.any():
       print(f"Constant features detected: {constant_features[constant_features].index.tolist()}")
       # Remove constant features
       df = df.loc[:, ~constant_features]
   ```

## See Also

- [Configuration Module](config.md) - Path and settings management
- [Environment Module](environments.md) - Using processed data in RL environment
- [Training Module](modeling.md) - Training with processed data
- [Getting Started Guide](../getting-started.md) - Complete setup workflow

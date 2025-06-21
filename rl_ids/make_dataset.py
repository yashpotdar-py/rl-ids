import os
from pathlib import Path
import time
from typing import Tuple

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
import typer

from rl_ids.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


class DataGenerator:
    """Handles loading and initial preprocessing of raw CSV data files"""

    def __init__(self):
        self.label_encoder = None
        self.processed_data = None

    def load_and_preprocess_data(self, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        """Load and preprocess CSV data files from the specified directory."""
        logger.info(f"🗂️  Starting data loading from directory: {data_dir}")

        # Validate input directory
        if not data_dir.exists():
            logger.error(f"❌ Data directory does not exist: {data_dir}")
            raise typer.Exit(1)

        # Find CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

        if not csv_files:
            logger.warning(f"⚠️  No CSV files found in directory: {data_dir}")
            return pd.DataFrame()

        logger.info(f"📄 Found {len(csv_files)} CSV files to process")

        # Load CSV files with progress tracking
        data_frames = []
        total_rows = 0

        for file in tqdm(csv_files, desc="Loading CSV files", unit="file"):
            file_path = os.path.join(data_dir, file)
            logger.debug(f"Loading file: {file}")

            try:
                df = pd.read_csv(file_path, low_memory=False)
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                data_frames.append(df)
                total_rows += len(df)
                logger.debug(
                    f"✅ Successfully loaded {file} with {len(df):,} rows")
            except Exception as e:
                logger.error(f"❌ Failed to load {file}: {str(e)}")
                continue

        if not data_frames:
            logger.error("❌ No data frames were successfully loaded")
            raise typer.Exit(1)

        # Concatenate data frames
        logger.info("🔗 Concatenating data frames...")
        with tqdm(total=1, desc="Combining datasets", unit="step") as pbar:
            data = pd.concat(data_frames, ignore_index=True)
            pbar.update(1)

        logger.info(
            f"📊 Combined dataset shape: {data.shape} ({len(data):,} total rows)")

        # Data preprocessing steps
        logger.info("🧹 Starting data preprocessing...")
        data = self._preprocess_data(data)

        # Store processed data
        self.processed_data = data

        logger.success(
            f"✅ Data loading and preprocessing complete! Final shape: {data.shape}")
        return data

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal preprocessing steps"""

        with tqdm(total=6, desc="Preprocessing steps", unit="step") as pbar:
            # Strip column names again after concatenation (safety measure)
            data.columns = data.columns.str.strip()
            logger.debug("🧹 Stripped whitespace from column names")
            pbar.update(1)

            # Remove columns with all null values
            initial_cols = data.shape[1]
            data.dropna(axis=1, how="all", inplace=True)
            removed_cols = initial_cols - data.shape[1]
            if removed_cols > 0:
                logger.info(
                    f"🗑️  Removed {removed_cols} columns with all null values")
            pbar.update(1)

            # Fill remaining null values
            null_count = data.isnull().sum().sum()
            if null_count > 0:
                logger.info(f"🔧 Filling {null_count:,} null values with 0")
                data.fillna(0, inplace=True)
            pbar.update(1)

            # Handle infinite values
            inf_count = np.isinf(data.select_dtypes(
                include=[np.number])).sum().sum()
            if inf_count > 0:
                logger.info(f"♾️  Replacing {inf_count:,} infinite values")
                data.replace([np.inf, -np.inf], 0, inplace=True)
            pbar.update(1)

            # Encode labels
            if "Label" in data.columns:
                logger.info("🏷️  Encoding labels...")
                # Keep original label column
                data["Label_Original"] = data["Label"].copy()
                # Encode the Label column
                self.label_encoder = LabelEncoder()
                data["Label"] = self.label_encoder.fit_transform(data["Label"])
                unique_labels = len(self.label_encoder.classes_)
                logger.info(f"📊 Encoded {unique_labels} unique labels")
                logger.info(
                    "💾 Original labels preserved in 'Label_Original' column")

                # Log label distribution
                label_dist = data['Label'].value_counts().sort_index()
                logger.info("📈 Label distribution:")
                for label, count in label_dist.items():
                    original_label = self.label_encoder.inverse_transform([label])[
                        0]
                    percentage = count / len(data) * 100
                    logger.info(
                        f"   {label} ({original_label}): {count:,} samples ({percentage:.1f}%)")
            else:
                logger.warning("⚠️  No 'Label' column found in dataset")
            pbar.update(1)

            # Data type optimization
            logger.info("⚡ Optimizing data types...")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Label', 'Label_Original']:
                    # Convert to float32 to save memory
                    data[col] = data[col].astype(np.float32)
            pbar.update(1)

        return data

    def get_label_mapping(self) -> dict:
        """Get mapping between encoded labels and original labels"""
        if self.label_encoder is None:
            return {}

        encoded_labels = range(len(self.label_encoder.classes_))
        original_labels = self.label_encoder.classes_
        return dict(zip(encoded_labels, original_labels))

    def save_processed_data(self, output_path: Path, data: pd.DataFrame = None) -> Path:
        """Save processed data to specified path"""
        if data is None:
            data = self.processed_data

        if data is None or data.empty:
            logger.error("❌ No data to save")
            raise ValueError("No processed data available")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving processed data to: {output_path}")
        try:
            data.to_csv(output_path, index=False)
            file_size = output_path.stat().st_size / 1024**2
            logger.success(
                f"✅ Successfully saved {len(data):,} rows to {output_path} ({file_size:.1f} MB)")
            return output_path
        except Exception as e:
            logger.error(f"❌ Failed to save processed data: {str(e)}")
            raise


class DataProcessor:
    """Enhanced data processing with multiple normalization and balancing strategies"""

    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self.label_mapping = None
        # Configure tqdm for better display
        tqdm.pandas(desc="Processing", dynamic_ncols=True)

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'robust',
        handle_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Improved normalization with outlier handling

        Args:
            df: Input dataframe
            method: Normalization method ('standard', 'robust', 'minmax')
            handle_outliers: Whether to handle outliers before normalization
        """
        # Validate method parameter
        valid_methods = ['standard', 'robust', 'minmax']
        if method not in valid_methods:
            raise ValueError(
                f"Method must be one of {valid_methods}, got '{method}'")

        logger.info(
            f"🔄 Starting data normalization using {method.upper()} scaler")
        logger.info(f"📊 Input shape: {df.shape}")

        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in [
            'Label', 'Label_Original']]
        X = df[feature_cols].copy()
        y = df[['Label', 'Label_Original']].copy()

        logger.info(f"📈 Processing {len(feature_cols)} features")

        # Handle infinite and NaN values
        logger.info("🧹 Cleaning infinite and NaN values...")
        with tqdm(total=3, desc="Data cleaning", unit="step") as pbar:
            # Replace infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            pbar.update(1)

            # Count and log NaN values
            nan_counts = X.isnull().sum()
            total_nans = nan_counts.sum()
            if total_nans > 0:
                logger.warning(
                    f"⚠️  Found {total_nans:,} NaN values across {(nan_counts > 0).sum()} columns")
                for col, count in nan_counts[nan_counts > 0].items():
                    logger.debug(f"   {col}: {count:,} NaN values")
            pbar.update(1)

            # Fill NaN with median (more robust than 0)
            X = X.fillna(X.median())
            logger.info("✅ NaN values filled with column medians")
            pbar.update(1)

        # Handle outliers if requested
        if handle_outliers:
            logger.info("🎯 Handling outliers...")
            X = self._handle_outliers(X)

        # Choose and apply scaler
        logger.info(f"⚙️  Applying {method.upper()} normalization...")
        if method == 'standard':
            self.scaler = StandardScaler()
            logger.debug("Using StandardScaler (mean=0, std=1)")
        elif method == 'robust':
            self.scaler = RobustScaler()
            logger.debug("Using RobustScaler (median and IQR based)")
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            logger.debug("Using MinMaxScaler (range [0,1])")

        # Fit and transform with progress
        with tqdm(total=2, desc="Scaling", unit="step") as pbar:
            self.scaler.fit(X)
            pbar.update(1)
            X_scaled = self.scaler.transform(X)
            pbar.update(1)

        X_scaled_df = pd.DataFrame(
            X_scaled, columns=feature_cols, index=X.index)

        # Store feature columns for later use
        self.feature_columns = feature_cols

        # Combine with labels
        df_normalized = pd.concat([X_scaled_df, y], axis=1)

        logger.success(
            f"✅ Normalization complete! Shape: {df_normalized.shape}")
        logger.info(
            f"📏 Feature range after scaling: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

        return df_normalized

    def _handle_outliers(self, X: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers using IQR method or percentile capping"""
        logger.info(
            f"🔍 Detecting and handling outliers using {method.upper()} method")

        outlier_counts = {}

        if method == 'iqr':
            # IQR method for each column with progress bar
            for col in tqdm(X.columns, desc="Processing columns", unit="col"):
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers before clipping
                outliers = ((X[col] < lower_bound) |
                            (X[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_counts[col] = outliers

                X[col] = X[col].clip(lower_bound, upper_bound)

        elif method == 'percentile':
            # Percentile capping (1st and 99th percentiles)
            for col in tqdm(X.columns, desc="Processing columns", unit="col"):
                lower_bound = X[col].quantile(0.01)
                upper_bound = X[col].quantile(0.99)

                # Count outliers before clipping
                outliers = ((X[col] < lower_bound) |
                            (X[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_counts[col] = outliers

                X[col] = X[col].clip(lower_bound, upper_bound)

        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            logger.info(
                f"📈 Handled {total_outliers:,} outliers across {len(outlier_counts)} columns")
            for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.debug(f"   {col}: {count:,} outliers")
        else:
            logger.info("✅ No outliers detected")

        return X

    def smart_balance_data(
        self,
        df: pd.DataFrame,
        strategy: str = 'smart_sampling',
        max_samples_per_class: int = 50000,
        min_samples_per_class: int = 1000
    ) -> pd.DataFrame:
        """
        Intelligent balancing strategy that doesn't create unnecessarily large datasets
        """
        # Validate strategy parameter
        valid_strategies = ['smart_sampling',
                            'smote', 'combined', 'hierarchical']
        if strategy not in valid_strategies:
            raise ValueError(
                f"Strategy must be one of {valid_strategies}, got '{strategy}'")

        logger.info(
            f"⚖️  Starting data balancing using {strategy.upper()} strategy")
        logger.info(
            f"🎯 Target: {min_samples_per_class:,} - {max_samples_per_class:,} samples per class")

        original_counts = df['Label'].value_counts().sort_index()
        logger.info(
            f"📊 Original distribution across {len(original_counts)} classes:")

        # Create a nice distribution table
        total_samples = len(df)
        for label, count in original_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(
                f"   Class {label}: {count:8,} samples ({percentage:5.1f}%)")

        logger.info(f"📈 Total samples: {total_samples:,}")
        logger.info(
            f"📊 Class imbalance ratio: {original_counts.max() / original_counts.min():.1f}:1")

        start_time = time.time()

        if strategy == 'smart_sampling':
            result = self._smart_sampling_balance(
                df, max_samples_per_class, min_samples_per_class)
        elif strategy == 'smote':
            result = self._smote_balance(df, max_samples_per_class)
        elif strategy == 'combined':
            result = self._combined_balance(df, max_samples_per_class)
        elif strategy == 'hierarchical':
            result = self._hierarchical_balance(
                df, max_samples_per_class, min_samples_per_class)

        duration = time.time() - start_time
        logger.success(f"✅ Balancing completed in {duration:.1f}s")

        # Log final distribution
        final_counts = result['Label'].value_counts().sort_index()
        final_total = len(result)

        logger.info("📊 Final distribution:")
        for label, count in final_counts.items():
            percentage = (count / final_total) * 100
            change = count - original_counts[label]
            change_symbol = "+" if change > 0 else ""
            logger.info(
                f"   Class {label}: {count:8,} samples ({percentage:5.1f}%) [{change_symbol}{change:,}]")

        logger.info(
            f"📈 Final total: {final_total:,} samples ({final_total/total_samples:.1f}x original)")

        return result

    def _smart_sampling_balance(self, df: pd.DataFrame, max_per_class: int, min_per_class: int) -> pd.DataFrame:
        """Smart sampling that creates reasonable dataset sizes while maintaining class balance"""
        logger.info("🧠 Using smart sampling strategy")

        class_counts = df['Label'].value_counts()
        majority_size = class_counts.max()

        # Calculate target sizes based on class hierarchy
        def calculate_target_size(current_size: int) -> int:
            if current_size >= majority_size * 0.1:  # Large classes
                return min(current_size, max_per_class)
            elif current_size >= 1000:  # Medium classes
                return min(max(current_size * 2, min_per_class), max_per_class // 2)
            elif current_size >= 100:  # Small classes
                return min(max(current_size * 5, min_per_class), max_per_class // 4)
            else:  # Very small classes
                return min(current_size * 10, min_per_class)

        balanced_dfs = []

        unique_labels = sorted(df['Label'].unique())
        for label in tqdm(unique_labels, desc="Balancing classes", unit="class"):
            class_df = df[df['Label'] == label]
            current_size = len(class_df)
            target_size = calculate_target_size(current_size)

            if target_size > current_size:
                # Oversample
                additional_samples = target_size - current_size
                oversampled = resample(
                    class_df,
                    n_samples=additional_samples,
                    random_state=42,
                    replace=True
                )
                combined_df = pd.concat(
                    [class_df, oversampled], ignore_index=True)
                action = f"oversampled (+{additional_samples:,})"
            elif target_size < current_size:
                # Undersample (for very large classes)
                combined_df = resample(
                    class_df,
                    n_samples=target_size,
                    random_state=42,
                    replace=False
                )
                action = f"undersampled (-{current_size - target_size:,})"
            else:
                combined_df = class_df
                action = "unchanged"

            balanced_dfs.append(combined_df)
            logger.debug(
                f"   Class {label}: {current_size:,} -> {len(combined_df):,} ({action})")

        logger.info("🔀 Shuffling balanced dataset...")
        result = pd.concat(balanced_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=42).reset_index(drop=True)

    def _smote_balance(self, df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
        """Use SMOTE for synthetic sample generation"""
        logger.info("🎨 Using SMOTE (Synthetic Minority Oversampling) strategy")

        X = df[self.feature_columns]
        y = df['Label']

        # Calculate sampling strategy to avoid huge datasets
        class_counts = y.value_counts()
        target_size = min(max_per_class, class_counts.median() * 3)

        sampling_strategy = {
            label: min(max_per_class, max(count, int(target_size)))
            for label, count in class_counts.items()
        }

        logger.info(
            f"🎯 SMOTE target sizes: {dict(sorted(sampling_strategy.items()))}")

        try:
            with tqdm(total=3, desc="SMOTE processing", unit="step") as pbar:
                smote = SMOTE(sampling_strategy=sampling_strategy,
                              random_state=42, k_neighbors=3)
                pbar.update(1)

                X_resampled, y_resampled = smote.fit_resample(X, y)
                pbar.update(1)

                result_df = pd.DataFrame(
                    X_resampled, columns=self.feature_columns)
                result_df['Label'] = y_resampled

                label_mapping = df.set_index(
                    'Label')['Label_Original'].drop_duplicates().to_dict()
                result_df['Label_Original'] = result_df['Label'].map(
                    label_mapping)
                pbar.update(1)

            logger.success("✅ SMOTE completed successfully")
            return result_df

        except Exception as e:
            logger.error(f"❌ SMOTE failed: {str(e)}")
            logger.info("🔄 Falling back to smart sampling strategy")
            return self._smart_sampling_balance(df, max_per_class, 1000)

    def _combined_balance(self, df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
        """Combined over/under sampling approach"""
        logger.info("🔄 Using combined SMOTE-Tomek strategy")

        X = df[self.feature_columns]
        y = df['Label']

        try:
            with tqdm(total=3, desc="Combined sampling", unit="step") as pbar:
                smote_tomek = SMOTETomek(
                    smote=SMOTE(random_state=42, k_neighbors=3),
                    random_state=42
                )
                pbar.update(1)

                X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
                pbar.update(1)

                result_df = pd.DataFrame(
                    X_resampled, columns=self.feature_columns)
                result_df['Label'] = y_resampled

                label_mapping = df.set_index(
                    'Label')['Label_Original'].drop_duplicates().to_dict()
                result_df['Label_Original'] = result_df['Label'].map(
                    label_mapping)
                pbar.update(1)

            logger.success("✅ Combined sampling completed successfully")
            return result_df

        except Exception as e:
            logger.error(f"❌ Combined sampling failed: {str(e)}")
            logger.info("🔄 Falling back to smart sampling strategy")
            return self._smart_sampling_balance(df, max_per_class, 1000)

    def _hierarchical_balance(self, df: pd.DataFrame, max_per_class: int, min_per_class: int) -> pd.DataFrame:
        """Hierarchical balancing based on class size tiers"""
        logger.info("🏗️  Using hierarchical balancing strategy")

        class_counts = df['Label'].value_counts().sort_values(ascending=False)

        # Define tiers
        tier_1_threshold = class_counts.iloc[0] * 0.1  # Large classes
        tier_2_threshold = class_counts.iloc[0] * 0.01  # Medium classes
        tier_3_threshold = 100  # Small classes

        logger.info("📊 Class tiers defined:")
        logger.info(f"   Tier 1 (Large): ≥{tier_1_threshold:,.0f} samples")
        logger.info(
            f"   Tier 2 (Medium): {tier_2_threshold:,.0f} - {tier_1_threshold:,.0f} samples")
        logger.info(
            f"   Tier 3 (Small): {tier_3_threshold} - {tier_2_threshold:,.0f} samples")
        logger.info(f"   Tier 4 (Tiny): <{tier_3_threshold} samples")

        balanced_dfs = []
        tier_stats = {"Large": 0, "Medium": 0, "Small": 0, "Tiny": 0}

        for label, count in tqdm(class_counts.items(), desc="Processing tiers", unit="class"):
            class_df = df[df['Label'] == label]

            if count >= tier_1_threshold:
                # Tier 1: Undersample large classes to reasonable size
                target_size = min(count, max_per_class)
                tier_name = "Large"
            elif count >= tier_2_threshold:
                # Tier 2: Keep as is or slight oversampling
                target_size = min(
                    max(count, min_per_class // 2), max_per_class // 2)
                tier_name = "Medium"
            elif count >= tier_3_threshold:
                # Tier 3: Moderate oversampling
                target_size = min(count * 3, max_per_class // 4)
                tier_name = "Small"
            else:
                # Tier 4: Aggressive oversampling for tiny classes
                target_size = min(count * 8, min_per_class)
                tier_name = "Tiny"

            tier_stats[tier_name] += 1

            if target_size > count:
                additional_samples = target_size - count
                oversampled = resample(
                    class_df, n_samples=additional_samples, random_state=42, replace=True)
                combined_df = pd.concat(
                    [class_df, oversampled], ignore_index=True)
            elif target_size < count:
                combined_df = resample(
                    class_df, n_samples=target_size, random_state=42, replace=False)
            else:
                combined_df = class_df

            balanced_dfs.append(combined_df)
            logger.debug(
                f"   Class {label} ({tier_name}): {count:,} -> {len(combined_df):,}")

        logger.info(f"📈 Tier distribution: {dict(tier_stats)}")

        result = pd.concat(balanced_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=42).reset_index(drop=True)

    def smart_split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Smart data splitting with stratification and size validation"""
        logger.info("✂️  Starting stratified data splitting")
        logger.info(
            f"🎯 Split ratios - Test: {test_size:.1%}, Validation: {val_size:.1%}, Train: {1-test_size-val_size:.1%}")

        with tqdm(total=4, desc="Data splitting", unit="step") as pbar:
            # Use StratifiedShuffleSplit for better control
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            pbar.update(1)

            train_idx, test_idx = next(splitter.split(df, df['Label']))
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            pbar.update(1)

            # Split train into train and validation
            if val_size > 0:
                val_relative_size = val_size / (1 - test_size)
                val_splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=val_relative_size,
                    random_state=random_state
                )

                train_idx, val_idx = next(
                    val_splitter.split(train_df, train_df['Label']))
                val_df = train_df.iloc[val_idx]
                train_df = train_df.iloc[train_idx]
            else:
                val_df = pd.DataFrame()
            pbar.update(1)

            # Reset indices
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(
                drop=True) if not val_df.empty else val_df
            test_df = test_df.reset_index(drop=True)
            pbar.update(1)

        logger.info("📊 Split results:")
        logger.info(
            f"   🏋️  Train set: {train_df.shape[0]:,} samples ({train_df.shape[0]/len(df):.1%})")
        if not val_df.empty:
            logger.info(
                f"   ✅ Validation set: {val_df.shape[0]:,} samples ({val_df.shape[0]/len(df):.1%})")
        logger.info(
            f"   🧪 Test set: {test_df.shape[0]:,} samples ({test_df.shape[0]/len(df):.1%})")

        # Verify stratification worked
        train_dist = train_df['Label'].value_counts(
            normalize=True).sort_index()
        test_dist = test_df['Label'].value_counts(normalize=True).sort_index()
        max_diff = abs(train_dist - test_dist).max()
        logger.info(
            f"📈 Stratification quality: max distribution difference = {max_diff:.3f}")

        return train_df, val_df, test_df


@app.command()
def main(
    raw_dir: Path = typer.Option(RAW_DATA_DIR, help="Raw data directory"),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, help="Output directory for splits"),
    normalize_method: str = typer.Option(
        'robust', help="Normalization method: standard, robust, or minmax"),
    balance_strategy: str = typer.Option(
        'smart_sampling', help="Balancing strategy: smart_sampling, smote, combined, or hierarchical"),
    max_samples_per_class: int = 30000,
    min_samples_per_class: int = 1000,
    test_size: float = 0.2,
    val_size: float = 0.1,
    handle_outliers: bool = True,
    save_intermediate: bool = False
):
    """
    Complete data processing pipeline: Load → Preprocess → Normalize → Balance → Split
    """
    start_time = time.time()
    logger.info("🚀 Starting complete data processing pipeline")
    logger.info("=" * 70)

    # 1. Generate/Load raw data
    logger.info("1️⃣  DATA GENERATION PHASE")
    generator = DataGenerator()
    df_raw = generator.load_and_preprocess_data(raw_dir)

    if df_raw.empty:
        logger.error("❌ No data generated - pipeline failed")
        raise typer.Exit(1)

    # Save intermediate processed data if requested
    if save_intermediate:
        processed_file = output_dir / "cicids2017_processed.csv"
        generator.save_processed_data(processed_file, df_raw)

    # 2. Initialize processor
    processor = DataProcessor()

    # 3. Normalize
    logger.info("\n2️⃣  NORMALIZATION PHASE")
    df_normalized = processor.normalize_data(
        df_raw,
        method=normalize_method,
        handle_outliers=handle_outliers
    )

    # 4. Balance
    logger.info("\n3️⃣  BALANCING PHASE")
    df_balanced = processor.smart_balance_data(
        df_normalized,
        strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
        min_samples_per_class=min_samples_per_class
    )

    # 5. Split data
    logger.info("\n4️⃣  SPLITTING PHASE")
    train_df, val_df, test_df = processor.smart_split_data(
        df_balanced,
        test_size=test_size,
        val_size=val_size
    )

    # 6. Save results
    logger.info("\n5️⃣  SAVING PHASE")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_tasks = [
        (train_df, "train.csv", "🏋️  Training data"),
        (test_df, "test.csv", "🧪 Test data"),
    ]

    if not val_df.empty:
        save_tasks.append((val_df, "val.csv", "✅ Validation data"))

    for data, filename, description in tqdm(save_tasks, desc="Saving files", unit="file"):
        file_path = output_dir / filename
        data.to_csv(file_path, index=False)
        file_size = file_path.stat().st_size / 1024**2
        logger.success(
            f"💾 {description} saved to {file_path} ({file_size:.1f} MB)")

    # Final summary
    total_time = time.time() - start_time
    total_samples = len(df_balanced)

    logger.info("\n" + "🎉 PROCESSING COMPLETE!" + "\n" + "=" * 60)
    logger.info(f"⏱️  Total processing time: {total_time:.1f} seconds")
    logger.info(f"📊 Original samples: {len(df_raw):,}")
    logger.info(
        f"📈 Final samples: {total_samples:,} ({total_samples/len(df_balanced):.1f}x)")
    logger.info(f"🔧 Normalization method: {normalize_method.upper()}")
    logger.info(f"⚖️  Balancing strategy: {balance_strategy.upper()}")
    logger.info(f"🎯 Max samples per class: {max_samples_per_class:,}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(
        f"🚀 Processing speed: {total_samples/total_time:,.0f} samples/second")
    logger.info("=" * 60)

    # Class distribution summary
    final_counts = df_balanced['Label'].value_counts().sort_index()
    logger.info("📊 Final class distribution summary:")
    logger.info(f"   Classes: {len(final_counts)}")
    logger.info(f"   Min class size: {final_counts.min():,}")
    logger.info(f"   Max class size: {final_counts.max():,}")
    logger.info(f"   Mean class size: {final_counts.mean():,.0f}")
    logger.info(f"   Std class size: {final_counts.std():,.0f}")
    logger.info(
        f"   Balance ratio: {final_counts.max()/final_counts.min():.1f}:1")


if __name__ == "__main__":
    app()

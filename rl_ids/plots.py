from pathlib import Path
from typing import Optional, Union
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import typer

from rl_ids.config import FIGURES_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')


app = typer.Typer()

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class IDSPlotter:
    """Comprehensive plotting class for IDS RL model analysis"""

    def __init__(self, figures_dir: Path = FIGURES_DIR, dpi: int = 300):
        self.figures_dir = Path(figures_dir)
        self.dpi = dpi
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'accent': '#592941',
            'light': '#F2F2F2',
            'dark': '#1A1A1A'
        }

        self.class_colors = sns.color_palette("husl", 15)  # For 15 classes

    def plot_training_metrics(self, metrics_path: Union[str, Path],
                              save_name: str = "training_metrics_overview") -> None:
        """Plot comprehensive training metrics overview"""
        logger.info("📊 Creating training metrics plots...")

        try:
            df = pd.read_csv(metrics_path)
        except FileNotFoundError:
            logger.error(f"❌ Training metrics file not found: {metrics_path}")
            return

        fig = plt.figure(figsize=(20, 16))

        # 1. Training Progress Overview (2x2 grid)
        gs = fig.add_gridspec(4, 3, height_ratios=[
                              1, 1, 1, 1], width_ratios=[1, 1, 1])

        # Training Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        episodes = df['Episode']
        accuracy = df['Train_Accuracy']
        ax1.plot(episodes, accuracy,
                 color=self.colors['primary'], linewidth=2, alpha=0.8)
        ax1.fill_between(episodes, accuracy, alpha=0.3,
                         color=self.colors['primary'])
        ax1.axhline(y=accuracy.mean(), color=self.colors['warning'], linestyle='--',
                    label=f'Mean: {accuracy.mean():.4f}')
        ax1.axhline(y=accuracy.max(), color=self.colors['success'], linestyle=':',
                    label=f'Max: {accuracy.max():.4f}')
        ax1.set_title('Training Accuracy Progress',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training Reward
        ax2 = fig.add_subplot(gs[0, 1])
        reward = df['Train_Reward']
        ax2.plot(episodes, reward,
                 color=self.colors['secondary'], linewidth=2, alpha=0.8)
        ax2.fill_between(episodes, reward, alpha=0.3,
                         color=self.colors['secondary'])

        # Add moving average
        window = min(20, len(reward) // 10)
        if window > 1:
            reward_ma = reward.rolling(window=window, center=True).mean()
            ax2.plot(episodes, reward_ma, color=self.colors['dark'], linewidth=3,
                     label=f'MA({window})')

        ax2.axhline(y=reward.mean(), color=self.colors['warning'], linestyle='--',
                    label=f'Mean: {reward.mean():.2f}')
        ax2.set_title('Training Reward Progress',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Training Loss
        ax3 = fig.add_subplot(gs[0, 2])
        loss = df['Train_Loss']
        ax3.plot(episodes, loss,
                 color=self.colors['accent'], linewidth=2, alpha=0.8)
        ax3.fill_between(episodes, loss, alpha=0.3,
                         color=self.colors['accent'])

        # Add moving average for loss
        if window > 1:
            loss_ma = loss.rolling(window=window, center=True).mean()
            ax3.plot(episodes, loss_ma, color=self.colors['dark'], linewidth=3,
                     label=f'MA({window})')

        ax3.set_title('Training Loss Progress', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Log scale for loss

        # Learning Rate Schedule
        ax4 = fig.add_subplot(gs[1, 0])
        lr = df['Learning_Rate']
        ax4.plot(episodes, lr, color=self.colors['success'], linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # Episode Length Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        episode_lengths = df['Episode_Length']
        ax5.hist(episode_lengths, bins=30, alpha=0.7, color=self.colors['primary'],
                 edgecolor='black')
        ax5.axvline(episode_lengths.mean(), color=self.colors['warning'], linestyle='--',
                    label=f'Mean: {episode_lengths.mean():.1f}')
        ax5.axvline(episode_lengths.median(), color=self.colors['secondary'], linestyle=':',
                    label=f'Median: {episode_lengths.median():.1f}')
        ax5.set_title('Episode Length Distribution',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Steps per Episode')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Validation metrics if available
        if 'Val_Accuracy' in df.columns:
            ax6 = fig.add_subplot(gs[1, 2])
            val_acc = df['Val_Accuracy']
            train_acc_resampled = df['Train_Accuracy']

            ax6.plot(episodes, train_acc_resampled, color=self.colors['primary'],
                     linewidth=2, label='Training', alpha=0.8)
            ax6.plot(episodes, val_acc, color=self.colors['secondary'],
                     linewidth=2, label='Validation', alpha=0.8)
            ax6.fill_between(episodes, train_acc_resampled, val_acc,
                             alpha=0.2, color=self.colors['accent'])
            ax6.set_title('Training vs Validation Accuracy',
                          fontsize=14, fontweight='bold')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Accuracy')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # Training Stability Analysis
        ax7 = fig.add_subplot(gs[2, :])

        # Calculate rolling statistics
        window_size = max(10, len(episodes) // 20)
        rolling_mean = accuracy.rolling(window=window_size, center=True).mean()
        rolling_std = accuracy.rolling(window=window_size, center=True).std()

        ax7.plot(episodes, accuracy, alpha=0.3,
                 color=self.colors['primary'], label='Raw Accuracy')
        ax7.plot(episodes, rolling_mean,
                 color=self.colors['primary'], linewidth=3, label=f'Rolling Mean ({window_size})')
        ax7.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                         alpha=0.2, color=self.colors['primary'], label='±1 Std Dev')

        ax7.set_title('Training Stability Analysis',
                      fontsize=14, fontweight='bold')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Accuracy')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Performance Summary Table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')

        # Calculate summary statistics
        final_accuracy = accuracy.iloc[-20:].mean() if len(
            accuracy) >= 20 else accuracy.mean()
        max_accuracy = accuracy.max()
        accuracy_std = accuracy.std()
        final_reward = reward.iloc[-20:].mean() if len(
            reward) >= 20 else reward.mean()

        summary_data = [
            ['Metric', 'Value', 'Description'],
            ['Final Accuracy (last 20)', f'{final_accuracy:.4f}',
             'Average of final 20 episodes'],
            ['Maximum Accuracy', f'{max_accuracy:.4f}',
                'Best accuracy achieved'],
            ['Accuracy Std Dev', f'{accuracy_std:.4f}',
                'Training stability measure'],
            ['Final Reward (last 20)', f'{final_reward:.2f}',
             'Average reward of final 20 episodes'],
            ['Total Episodes', f'{len(episodes)}', 'Training duration'],
            ['Final Learning Rate', f'{lr.iloc[-1]:.2e}', 'LR at training end']
        ]

        table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0],
                          cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

        plt.suptitle('Training Metrics Comprehensive Analysis',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"📊 Training metrics plot saved: {save_path}")

    def plot_evaluation_overview(self, reports_dir: Path,
                                 save_name: str = "evaluation_overview") -> None:
        """Plot comprehensive evaluation overview"""
        logger.info("📊 Creating evaluation overview plots...")

        # Load evaluation data
        try:
            summary_df = pd.read_csv(
                reports_dir / "evaluation_summary_enhanced.csv")
            episode_details = pd.read_csv(
                reports_dir / "evaluation_episode_details_enhanced.csv")
            predictions_df = pd.read_csv(
                reports_dir / "evaluation_detailed_predictions_enhanced.csv")

        except FileNotFoundError as e:
            logger.error(f"❌ Evaluation file not found: {e}")
            return

        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1])

        # 1. Episode Performance Trends
        ax1 = fig.add_subplot(gs[0, :2])
        episodes = episode_details['episode']
        accuracy = episode_details['accuracy']
        confidence = episode_details['avg_confidence']

        ax1_twin = ax1.twinx()

        line1 = ax1.plot(episodes, accuracy, 'o-', color=self.colors['primary'],
                         linewidth=2, markersize=6, label='Accuracy')
        line2 = ax1_twin.plot(episodes, confidence, 's-', color=self.colors['secondary'],
                              linewidth=2, markersize=6, label='Avg Confidence')

        ax1.axhline(y=accuracy.mean(),
                    color=self.colors['primary'], linestyle='--', alpha=0.7)
        ax1_twin.axhline(
            y=confidence.mean(), color=self.colors['secondary'], linestyle='--', alpha=0.7)

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy', color=self.colors['primary'])
        ax1_twin.set_ylabel('Average Confidence',
                            color=self.colors['secondary'])
        ax1.set_title('Episode-wise Performance Trends',
                      fontsize=14, fontweight='bold')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Confidence Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        confidence_scores = predictions_df['Confidence']
        ax2.hist(confidence_scores, bins=50, alpha=0.7, color=self.colors['accent'],
                 edgecolor='black', density=True)
        ax2.axvline(confidence_scores.mean(), color=self.colors['warning'], linestyle='--',
                    linewidth=2, label=f'Mean: {confidence_scores.mean():.3f}')
        ax2.axvline(0.8, color=self.colors['success'], linestyle=':',
                    linewidth=2, label='Threshold: 0.8')
        ax2.set_title('Confidence Score Distribution',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Prediction Time Analysis
        ax3 = fig.add_subplot(gs[0, 3])
        pred_times = episode_details['avg_prediction_time_ms']
        ax3.boxplot(pred_times, patch_artist=True,
                    boxprops=dict(facecolor=self.colors['primary'], alpha=0.7))
        ax3.set_title('Prediction Time Distribution',
                      fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        ax3.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"""Mean: {pred_times.mean():.2f}ms
Median: {pred_times.median():.2f}ms
Std: {pred_times.std():.2f}ms
Min: {pred_times.min():.2f}ms
Max: {pred_times.max():.2f}ms"""
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Class Distribution Analysis
        ax4 = fig.add_subplot(gs[1, :2])
        true_labels = predictions_df['True_Label']
        pred_labels = predictions_df['Predicted_Label']

        unique_labels = sorted(true_labels.unique())
        true_counts = true_labels.value_counts().sort_index()
        pred_counts = pred_labels.value_counts().reindex(unique_labels, fill_value=0)

        x = np.arange(len(unique_labels))
        width = 0.35

        bars1 = ax4.bar(x - width/2, true_counts.values, width,
                        label='True Distribution', alpha=0.8, color=self.colors['primary'])
        bars2 = ax4.bar(x + width/2, pred_counts.values, width,
                        label='Predicted Distribution', alpha=0.8, color=self.colors['secondary'])

        ax4.set_xlabel('Class Label')
        ax4.set_ylabel('Count')
        ax4.set_title('True vs Predicted Class Distribution',
                      fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Class_{i}' for i in unique_labels], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Accuracy by Confidence Level
        ax5 = fig.add_subplot(gs[1, 2])

        # Bin confidence scores
        confidence_bins = np.linspace(0, 1, 11)
        bin_labels = [f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}'
                      for i in range(len(confidence_bins)-1)]

        predictions_df['confidence_bin'] = pd.cut(predictions_df['Confidence'],
                                                  bins=confidence_bins, labels=bin_labels)

        bin_accuracy = predictions_df.groupby(
            'confidence_bin')['Correct'].mean()
        bin_counts = predictions_df.groupby('confidence_bin').size()

        bars = ax5.bar(range(len(bin_accuracy)), bin_accuracy.values,
                       alpha=0.7, color=self.colors['success'])

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, bin_counts)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'n={count}', ha='center', va='bottom', fontsize=8)

        ax5.set_xlabel('Confidence Bin')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Accuracy by Confidence Level',
                      fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(bin_labels)))
        ax5.set_xticklabels(bin_labels, rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. Error Analysis
        ax6 = fig.add_subplot(gs[1, 3])
        correct_mask = predictions_df['Correct']

        # Confidence comparison for correct vs incorrect
        correct_conf = predictions_df[correct_mask]['Confidence']
        incorrect_conf = predictions_df[~correct_mask]['Confidence']

        ax6.hist([correct_conf, incorrect_conf], bins=30, alpha=0.7,
                 label=['Correct', 'Incorrect'], color=[self.colors['success'], self.colors['warning']])
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Count')
        ax6.set_title('Confidence: Correct vs Incorrect',
                      fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Confusion Matrix Heatmap
        ax7 = fig.add_subplot(gs[2, :2])
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax7.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax7.figure.colorbar(im, ax=ax7)

        ax7.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=[f'C{i}' for i in unique_labels],
                yticklabels=[f'C{i}' for i in unique_labels],
                title="Normalized Confusion Matrix",
                ylabel='True label',
                xlabel='Predicted label')

        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax7.text(j, i, f'{cm_normalized[i, j]:.2f}',
                         ha="center", va="center",
                         color="white" if cm_normalized[i,
                                                        j] > thresh else "black",
                         fontsize=8)
        plt.setp(ax7.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        # 8. Performance Metrics Radar Chart
        ax8 = fig.add_subplot(gs[2, 2], projection='polar')

        # Get metrics from summary
        metrics = ['overall_accuracy',
                   'high_confidence_accuracy', 'average_confidence']
        if 'macro_avg_precision' in summary_df.columns:
            metrics.extend(
                ['macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1_score'])

        values = [summary_df[metric].iloc[0] for metric in metrics]
        labels = [metric.replace('_', ' ').title() for metric in metrics]

        # Close the plot
        values += values[:1]
        labels += labels[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

        ax8.plot(angles, values, 'o-', linewidth=2,
                 color=self.colors['primary'])
        ax8.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(labels[:-1], fontsize=10)
        ax8.set_ylim(0, 1)
        ax8.set_title('Performance Metrics Overview',
                      fontsize=14, fontweight='bold', pad=20)
        ax8.grid(True)

        # 9. Key Statistics Summary
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.axis('off')

        # Calculate key statistics
        total_predictions = len(predictions_df)
        overall_accuracy = summary_df['overall_accuracy'].iloc[0]
        high_conf_accuracy = summary_df['high_confidence_accuracy'].iloc[0]
        high_conf_percentage = summary_df['high_confidence_percentage'].iloc[0]
        avg_pred_time = summary_df['avg_prediction_time_ms'].iloc[0]

        stats_text = f"""
📊 EVALUATION SUMMARY
{'='*25}

Total Predictions: {total_predictions:,}
Overall Accuracy: {overall_accuracy:.4f}
High-Conf Accuracy: {high_conf_accuracy:.4f}
High-Conf %: {high_conf_percentage:.1f}%
Avg Pred Time: {avg_pred_time:.2f}ms

Test Episodes: {len(episode_details)}
Unique Classes: {len(unique_labels)}
Confidence Threshold: 0.8
"""

        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['light'], alpha=0.8))

        # 10. Class-wise Performance Metrics
        ax10 = fig.add_subplot(gs[3, :])

        # Calculate per-class metrics
        class_metrics = []
        for label in unique_labels:
            mask = true_labels == label
            if mask.sum() > 0:
                class_pred = pred_labels[mask]
                class_correct = (class_pred == label).sum()
                class_total = mask.sum()
                class_acc = class_correct / class_total

                # Average confidence for this class
                class_conf = predictions_df[mask]['Confidence'].mean()

                class_metrics.append({
                    'class': f'Class_{label}',
                    'accuracy': class_acc,
                    'confidence': class_conf,
                    'support': class_total
                })

        class_df = pd.DataFrame(class_metrics)

        # Create grouped bar chart
        x = np.arange(len(class_df))
        width = 0.35

        bars1 = ax10.bar(x - width/2, class_df['accuracy'], width,
                         label='Accuracy', alpha=0.8, color=self.colors['primary'])
        bars2 = ax10.bar(x + width/2, class_df['confidence'], width,
                         label='Avg Confidence', alpha=0.8, color=self.colors['secondary'])

        # Add support count on top of accuracy bars
        for i, (bar, support) in enumerate(zip(bars1, class_df['support'])):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'n={support}', ha='center', va='bottom', fontsize=8)

        ax10.set_xlabel('Class')
        ax10.set_ylabel('Score')
        ax10.set_title('Per-Class Performance Metrics',
                       fontsize=14, fontweight='bold')
        ax10.set_xticks(x)
        ax10.set_xticklabels(class_df['class'], rotation=45)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(0, 1.1)

        plt.suptitle('Comprehensive Evaluation Analysis',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"📊 Evaluation overview plot saved: {save_path}")

    def plot_class_analysis(self, reports_dir: Path,
                            save_name: str = "class_analysis") -> None:
        """Detailed per-class analysis plots"""
        logger.info("📊 Creating detailed class analysis plots...")

        try:
            predictions_df = pd.read_csv(
                reports_dir / "evaluation_detailed_predictions_enhanced.csv")
            classification_report_df = pd.read_csv(
                reports_dir / "evaluation_classification_report.csv")
        except FileNotFoundError as e:
            logger.error(f"❌ Required file not found: {e}")
            return

        unique_labels = sorted(predictions_df['True_Label'].unique())
        n_classes = len(unique_labels)

        # Create comprehensive class analysis
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1])

        # 1. Per-class confusion matrix
        ax1 = fig.add_subplot(gs[0, :])
        cm = confusion_matrix(predictions_df['True_Label'], predictions_df['Predicted_Label'],
                              labels=unique_labels)

        # Create a more detailed heatmap
        mask = cm == 0
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', mask=mask,
                    xticklabels=[f'C{i}' for i in unique_labels],
                    yticklabels=[f'C{i}' for i in unique_labels],
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Detailed Confusion Matrix (Raw Counts)',
                      fontsize=16, fontweight='bold')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')

        # 2. Class-wise metrics comparison
        ax2 = fig.add_subplot(gs[1, :])

        # Debug: Check the classification report structure
        logger.debug(
            f"Classification report columns: {classification_report_df.columns.tolist()}")
        logger.debug(
            f"Classification report index: {classification_report_df.index.tolist()}")

        # Set the first column as index if it contains the class names
        if 'Unnamed: 0' in classification_report_df.columns:
            classification_report_df = classification_report_df.set_index(
                'Unnamed: 0')

        # Extract metrics from classification report
        metrics_data = []
        for label in unique_labels:
            # Try different possible class name formats
            possible_names = [
                f'Class_{label}',
                str(label),
                # Also check for original class names in the index
            ]

            # If we have True_Class in predictions_df, get the original name
            if 'True_Class' in predictions_df.columns:
                class_mask = predictions_df['True_Label'] == label
                if class_mask.sum() > 0:
                    original_name = predictions_df[class_mask]['True_Class'].iloc[0]
                    possible_names.append(original_name)

            row_data = None
            class_name_used = None

            for name in possible_names:
                if name in classification_report_df.index:
                    row_data = classification_report_df.loc[name]
                    class_name_used = name
                    break

            if row_data is not None:
                try:
                    metrics_data.append({
                        'class': f'Class_{label}',
                        'original_name': class_name_used,
                        'precision': float(row_data['precision']) if 'precision' in row_data else 0.0,
                        'recall': float(row_data['recall']) if 'recall' in row_data else 0.0,
                        'f1-score': float(row_data['f1-score']) if 'f1-score' in row_data else 0.0,
                        'support': float(row_data['support']) if 'support' in row_data else 0.0
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Could not extract metrics for class {label}: {e}")
                    # Calculate metrics manually from predictions
                    class_mask = predictions_df['True_Label'] == label
                    if class_mask.sum() > 0:
                        class_predictions = predictions_df[class_mask]
                        accuracy = class_predictions['Correct'].mean()
                        support = len(class_predictions)

                        metrics_data.append({
                            'class': f'Class_{label}',
                            'original_name': class_name_used or f'Class_{label}',
                            'precision': accuracy,  # Simplified - using accuracy as proxy
                            'recall': accuracy,
                            'f1-score': accuracy,
                            'support': support
                        })

        if not metrics_data:
            logger.warning(
                "No metrics data found, calculating from predictions...")
            # Fallback: calculate metrics directly from predictions
            for label in unique_labels:
                class_mask = predictions_df['True_Label'] == label
                if class_mask.sum() > 0:
                    class_predictions = predictions_df[class_mask]
                    accuracy = class_predictions['Correct'].mean()
                    support = len(class_predictions)

                    # Calculate precision and recall manually
                    true_positives = class_predictions['Correct'].sum()
                    false_negatives = len(class_predictions) - true_positives

                    # For predicted as this class
                    pred_mask = predictions_df['Predicted_Label'] == label
                    pred_correct = predictions_df[pred_mask]['Correct'].sum()
                    false_positives = pred_mask.sum() - pred_correct

                    precision = true_positives / \
                        (true_positives + false_positives) if (true_positives +
                                                               false_positives) > 0 else 0
                    recall = true_positives / \
                        (true_positives + false_negatives) if (true_positives +
                                                               false_negatives) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0

                    metrics_data.append({
                        'class': f'Class_{label}',
                        'original_name': f'Class_{label}',
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1,
                        'support': support
                    })

        metrics_df = pd.DataFrame(metrics_data)

        if len(metrics_df) == 0:
            logger.error("❌ Could not extract any class metrics")
            plt.close(fig)
            return

        x = np.arange(len(metrics_df))
        width = 0.25

        ax2.bar(x - width, metrics_df['precision'], width, label='Precision',
                alpha=0.8, color=self.colors['primary'])
        ax2.bar(x, metrics_df['recall'], width, label='Recall',
                alpha=0.8, color=self.colors['secondary'])
        ax2.bar(x + width, metrics_df['f1-score'], width, label='F1-Score',
                alpha=0.8, color=self.colors['success'])

        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Performance Metrics Comparison',
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df['class'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)

        # 3. Support distribution
        ax3 = fig.add_subplot(gs[2, 0])
        support_values = metrics_df['support'].values
        bars = ax3.bar(range(len(support_values)), support_values,
                       color=self.class_colors[:len(support_values)], alpha=0.7)

        # Add value labels on bars
        for bar, val in zip(bars, support_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(support_values)*0.01,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10)

        ax3.set_title('Class Support Distribution',
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('Class Index')
        ax3.set_ylabel('Number of Samples')
        ax3.set_xticks(range(len(metrics_df)))
        ax3.set_xticklabels([f'C{i}' for i in unique_labels], rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Precision-Recall scatter
        ax4 = fig.add_subplot(gs[2, 1])
        scatter = ax4.scatter(metrics_df['recall'], metrics_df['precision'],
                              s=metrics_df['support'] /
                              max(metrics_df['support'])*500,
                              c=range(len(metrics_df)), cmap='viridis', alpha=0.7)

        # Add class labels
        for i, row in metrics_df.iterrows():
            ax4.annotate(f'C{unique_labels[i]}',
                         (row['recall'], row['precision']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Analysis\n(Bubble size = Support)',
                      fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1.05)
        ax4.set_ylim(0, 1.05)

        # Add diagonal line
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Balance')
        ax4.legend()

        # 5. F1-Score ranking
        ax5 = fig.add_subplot(gs[2, 2])
        sorted_metrics = metrics_df.sort_values('f1-score', ascending=True)

        bars = ax5.barh(range(len(sorted_metrics)), sorted_metrics['f1-score'],
                        color=self.class_colors[:len(sorted_metrics)], alpha=0.7)

        ax5.set_yticks(range(len(sorted_metrics)))
        ax5.set_yticklabels(sorted_metrics['class'])
        ax5.set_xlabel('F1-Score')
        ax5.set_title('F1-Score Ranking', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_metrics['f1-score'])):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', ha='left', va='center', fontsize=9)

        # 6-8. Confidence analysis per class (3 subplots)
        for idx, (start_class, end_class) in enumerate([(0, 5), (5, 10), (10, len(unique_labels))]):
            ax = fig.add_subplot(gs[3+idx, :])
            class_subset = unique_labels[start_class:end_class]

            confidence_data = []
            labels_for_plot = []

            for class_label in class_subset:
                class_mask = predictions_df['True_Label'] == class_label
                if class_mask.sum() > 0:  # Only include if we have data
                    class_confidences = predictions_df[class_mask]['Confidence']
                    confidence_data.append(class_confidences)
                    labels_for_plot.append(f'Class_{class_label}')

            if confidence_data:  # Only plot if we have data
                bp = ax.boxplot(confidence_data,
                                labels=labels_for_plot, patch_artist=True)

                # Color the boxes
                for patch, color in zip(bp['boxes'], self.class_colors[start_class:start_class+len(confidence_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_title(f'Confidence Distribution - Classes {start_class}-{min(end_class-1, len(unique_labels)-1)}',
                             fontsize=14, fontweight='bold')
                ax.set_ylabel('Confidence Score')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.8, color='red', linestyle='--',
                           alpha=0.7, label='Threshold')
                if idx == 0:  # Only show legend on first subplot
                    ax.legend()
            else:
                # Empty subplot if no data
                ax.text(0.5, 0.5, f'No data for classes {start_class}-{end_class-1}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=14, style='italic')
                ax.set_title(f'Confidence Distribution - Classes {start_class}-{end_class-1}',
                             fontsize=14, fontweight='bold')

        plt.suptitle('Comprehensive Per-Class Analysis',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"📊 Class analysis plot saved: {save_path}")

    def plot_error_analysis(self, reports_dir: Path,
                            save_name: str = "error_analysis") -> None:
        """Detailed error analysis and misclassification patterns"""
        logger.info("📊 Creating error analysis plots...")

        try:
            predictions_df = pd.read_csv(
                reports_dir / "evaluation_detailed_predictions_enhanced.csv")
        except FileNotFoundError as e:
            logger.error(f"❌ Predictions file not found: {e}")
            return

        # Filter incorrect predictions
        incorrect_df = predictions_df[~predictions_df['Correct']].copy()

        if len(incorrect_df) == 0:
            logger.warning(
                "⚠️ No incorrect predictions found - perfect accuracy!")
            return

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

        # 1. Error distribution by true class
        ax1 = fig.add_subplot(gs[0, 0])
        error_by_true = incorrect_df['True_Label'].value_counts().sort_index()
        bars = ax1.bar(range(len(error_by_true)), error_by_true.values,
                       color=self.colors['warning'], alpha=0.7)

        ax1.set_title('Errors by True Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('True Class')
        ax1.set_ylabel('Number of Errors')
        ax1.set_xticks(range(len(error_by_true)))
        ax1.set_xticklabels([f'C{i}' for i in error_by_true.index])
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, error_by_true.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(error_by_true.values)*0.01,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10)

        # 2. Error distribution by predicted class
        ax2 = fig.add_subplot(gs[0, 1])
        error_by_pred = incorrect_df['Predicted_Label'].value_counts(
        ).sort_index()
        bars = ax2.bar(range(len(error_by_pred)), error_by_pred.values,
                       color=self.colors['secondary'], alpha=0.7)

        ax2.set_title('Errors by Predicted Class',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('Number of Errors')
        ax2.set_xticks(range(len(error_by_pred)))
        ax2.set_xticklabels([f'C{i}' for i in error_by_pred.index])
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, error_by_pred.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(error_by_pred.values)*0.01,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10)

        # 3. Confidence distribution for errors
        ax3 = fig.add_subplot(gs[0, 2])
        correct_conf = predictions_df[predictions_df['Correct']]['Confidence']
        incorrect_conf = incorrect_df['Confidence']

        ax3.hist([correct_conf, incorrect_conf], bins=30, alpha=0.7,
                 label=['Correct', 'Incorrect'],
                 color=[self.colors['success'], self.colors['warning']], density=True)

        ax3.axvline(correct_conf.mean(), color=self.colors['success'], linestyle='--',
                    label=f'Correct Mean: {correct_conf.mean():.3f}')
        ax3.axvline(incorrect_conf.mean(), color=self.colors['warning'], linestyle='--',
                    label=f'Incorrect Mean: {incorrect_conf.mean():.3f}')

        ax3.set_title('Confidence: Correct vs Incorrect',
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Most common misclassification patterns
        ax4 = fig.add_subplot(gs[1, :])

        # Create misclassification pairs
        misclass_pairs = incorrect_df.groupby(
            ['True_Label', 'Predicted_Label']).size()
        top_misclass = misclass_pairs.nlargest(15)

        pair_labels = [f'{true}→{pred}' for (true, pred) in top_misclass.index]

        bars = ax4.bar(range(len(top_misclass)), top_misclass.values,
                       color=self.colors['accent'], alpha=0.7)

        ax4.set_title('Top 15 Misclassification Patterns',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('True→Predicted Class')
        ax4.set_ylabel('Frequency')
        ax4.set_xticks(range(len(pair_labels)))
        ax4.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, top_misclass.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(top_misclass.values)*0.01,
                     f'{int(val)}', ha='center', va='bottom', fontsize=9)

        # 5. Error rate by confidence bins
        ax5 = fig.add_subplot(gs[2, 0])

        confidence_bins = np.linspace(0, 1, 11)
        bin_labels = [f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}'
                      for i in range(len(confidence_bins)-1)]

        predictions_df['confidence_bin'] = pd.cut(predictions_df['Confidence'],
                                                  bins=confidence_bins, labels=bin_labels)

        error_rate_by_bin = 1 - \
            predictions_df.groupby('confidence_bin')['Correct'].mean()
        bin_counts = predictions_df.groupby('confidence_bin').size()

        bars = ax5.bar(range(len(error_rate_by_bin)), error_rate_by_bin.values,
                       alpha=0.7, color=self.colors['warning'])

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, bin_counts)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'n={count}', ha='center', va='bottom', fontsize=8)

        ax5.set_title('Error Rate by Confidence Bin',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Confidence Bin')
        ax5.set_ylabel('Error Rate')
        ax5.set_xticks(range(len(bin_labels)))
        ax5.set_xticklabels(bin_labels, rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. High confidence errors analysis
        ax6 = fig.add_subplot(gs[2, 1])

        high_conf_errors = incorrect_df[incorrect_df['High_Confidence']]

        if len(high_conf_errors) > 0:
            high_conf_by_class = high_conf_errors['True_Label'].value_counts(
            ).sort_index()
            bars = ax6.bar(range(len(high_conf_by_class)), high_conf_by_class.values,
                           color=self.colors['accent'], alpha=0.7)

            ax6.set_title(f'High-Confidence Errors (≥0.8)\nTotal: {len(high_conf_errors)}',
                          fontsize=14, fontweight='bold')
            ax6.set_xlabel('True Class')
            ax6.set_ylabel('Count')
            ax6.set_xticks(range(len(high_conf_by_class)))
            ax6.set_xticklabels([f'C{i}' for i in high_conf_by_class.index])
            ax6.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, high_conf_by_class.values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + max(high_conf_by_class.values)*0.01,
                         f'{int(val)}', ha='center', va='bottom', fontsize=10)
        else:
            ax6.text(0.5, 0.5, 'No High-Confidence\nErrors Found!',
                     ha='center', va='center', transform=ax6.transAxes,
                     fontsize=16, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=self.colors['success'], alpha=0.3))
            ax6.set_title('High-Confidence Errors (≥0.8)',
                          fontsize=14, fontweight='bold')

        # 7. Confusion matrix of errors only
        ax7 = fig.add_subplot(gs[2, 2])

        unique_labels = sorted(predictions_df['True_Label'].unique())
        error_cm = confusion_matrix(incorrect_df['True_Label'], incorrect_df['Predicted_Label'],
                                    labels=unique_labels)

        # Only show non-zero entries
        mask = error_cm == 0
        im = ax7.imshow(error_cm, cmap='Reds', alpha=0.8)

        # Add text annotations for non-zero values
        for i in range(error_cm.shape[0]):
            for j in range(error_cm.shape[1]):
                if error_cm[i, j] > 0:
                    ax7.text(j, i, f'{error_cm[i, j]}', ha="center", va="center",
                             color="white" if error_cm[i, j] > error_cm.max(
                    )/2 else "black",
                        fontweight='bold')

        ax7.set_title('Error Confusion Matrix', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Predicted Class')
        ax7.set_ylabel('True Class')
        ax7.set_xticks(range(len(unique_labels)))
        ax7.set_xticklabels([f'C{i}' for i in unique_labels])
        ax7.set_yticks(range(len(unique_labels)))
        ax7.set_yticklabels([f'C{i}' for i in unique_labels])

        # 8. Error statistics summary
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')

        # Calculate comprehensive error statistics
        total_predictions = len(predictions_df)
        total_errors = len(incorrect_df)
        error_rate = total_errors / total_predictions

        high_conf_errors_count = len(
            incorrect_df[incorrect_df['High_Confidence']])
        high_conf_total = len(
            predictions_df[predictions_df['High_Confidence']])
        high_conf_error_rate = high_conf_errors_count / \
            high_conf_total if high_conf_total > 0 else 0

        # Most problematic classes
        error_rates_by_class = {}
        for label in unique_labels:
            class_mask = predictions_df['True_Label'] == label
            class_total = class_mask.sum()
            class_errors = (~predictions_df[class_mask]['Correct']).sum()
            error_rates_by_class[label] = class_errors / \
                class_total if class_total > 0 else 0

        worst_class = max(error_rates_by_class, key=error_rates_by_class.get)
        best_class = min(error_rates_by_class, key=error_rates_by_class.get)

        avg_conf_correct = correct_conf.mean()
        avg_conf_incorrect = incorrect_conf.mean()
        conf_difference = avg_conf_correct - avg_conf_incorrect

        summary_text = f"""
🔍 ERROR ANALYSIS SUMMARY
{'='*50}

Overall Statistics:
  • Total Predictions: {total_predictions:,}
  • Total Errors: {total_errors:,}
  • Overall Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)

High-Confidence Analysis:
  • High-Conf Predictions: {high_conf_total:,}
  • High-Conf Errors: {high_conf_errors_count:,}
  • High-Conf Error Rate: {high_conf_error_rate:.4f} ({high_conf_error_rate*100:.2f}%)

Class Performance:
  • Most Problematic: Class_{worst_class} ({error_rates_by_class[worst_class]*100:.2f}% error rate)
  • Best Performing: Class_{best_class} ({error_rates_by_class[best_class]*100:.2f}% error rate)

Confidence Analysis:
  • Avg Confidence (Correct): {avg_conf_correct:.4f}
  • Avg Confidence (Incorrect): {avg_conf_incorrect:.4f}
  • Confidence Difference: {conf_difference:.4f}

Top Misclassification: {pair_labels[0] if pair_labels else 'None'} ({top_misclass.iloc[0] if len(top_misclass) > 0 else 0} cases)
"""

        ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['light'], alpha=0.9))

        plt.suptitle('Comprehensive Error Analysis',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"📊 Error analysis plot saved: {save_path}")

    def create_all_plots(self, reports_dir: Path = REPORTS_DIR,
                         metrics_path: Optional[Path] = None) -> None:
        """Create all analysis plots"""
        logger.info("🎨 Creating comprehensive analysis plots...")

        if metrics_path is None:
            metrics_path = reports_dir / "training_metrics.csv"

        # Check if files exist
        if not reports_dir.exists():
            logger.error(f"❌ Reports directory not found: {reports_dir}")
            return

        # Create training plots if metrics available
        if metrics_path.exists():
            self.plot_training_metrics(metrics_path)
        else:
            logger.warning(f"⚠️ Training metrics not found: {metrics_path}")

        # Create evaluation plots
        required_files = [
            "evaluation_summary_enhanced.csv",
            "evaluation_detailed_predictions_enhanced.csv",
            "evaluation_episode_details_enhanced.csv"
        ]

        missing_files = [f for f in required_files if not (
            reports_dir / f).exists()]

        if missing_files:
            logger.warning(f"⚠️ Missing evaluation files: {missing_files}")
            logger.info(
                "💡 Run evaluation first: python -m rl_ids.modeling.evaluate")
        else:
            self.plot_evaluation_overview(reports_dir)
            self.plot_class_analysis(reports_dir)
            self.plot_error_analysis(reports_dir)

        logger.success(f"✨ All plots saved to: {self.figures_dir}")


# CLI Commands
@app.command()
def training_plots(
    metrics_path: Path = typer.Option(
        REPORTS_DIR / "training_metrics.csv",
        help="Path to training metrics CSV file"
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Output directory for figures"),
    dpi: int = typer.Option(300, help="Resolution for saved plots")
):
    """Generate training analysis plots"""
    plotter = IDSPlotter(figures_dir, dpi)
    plotter.plot_training_metrics(metrics_path)


@app.command()
def evaluation_plots(
    reports_dir: Path = typer.Option(REPORTS_DIR, help="Reports directory"),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Output directory for figures"),
    dpi: int = typer.Option(300, help="Resolution for saved plots")
):
    """Generate evaluation analysis plots"""
    plotter = IDSPlotter(figures_dir, dpi)
    plotter.plot_evaluation_overview(reports_dir)


@app.command()
def class_plots(
    reports_dir: Path = typer.Option(REPORTS_DIR, help="Reports directory"),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Output directory for figures"),
    dpi: int = typer.Option(300, help="Resolution for saved plots")
):
    """Generate detailed class analysis plots"""
    plotter = IDSPlotter(figures_dir, dpi)
    plotter.plot_class_analysis(reports_dir)


@app.command()
def error_plots(
    reports_dir: Path = typer.Option(REPORTS_DIR, help="Reports directory"),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Output directory for figures"),
    dpi: int = typer.Option(300, help="Resolution for saved plots")
):
    """Generate error analysis plots"""
    plotter = IDSPlotter(figures_dir, dpi)
    plotter.plot_error_analysis(reports_dir)


@app.command()
def all_plots(
    reports_dir: Path = typer.Option(REPORTS_DIR, help="Reports directory"),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Output directory for figures"),
    metrics_path: Optional[Path] = typer.Option(
        None, help="Training metrics path"),
    dpi: int = typer.Option(300, help="Resolution for saved plots")
):
    """Generate all analysis plots"""
    plotter = IDSPlotter(figures_dir, dpi)
    plotter.create_all_plots(reports_dir, metrics_path)


if __name__ == "__main__":
    app()

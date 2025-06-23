import os
from pathlib import Path
import time

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
from tqdm import tqdm
import typer

from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TEST_DATA_FILE, TRAIN_DATA_FILE
from rl_ids.environments.ids_env import IDSDetectionEnv

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "dqn_model_final.pt", help="Path to trained DQN model"
    ),
    test_data_path: Path = typer.Option(
        TEST_DATA_FILE, help="Path to test dataset"
    ),
    train_data_path: Path = typer.Option(
        TRAIN_DATA_FILE, help="Path to training dataset (for label mapping)"
    ),
    reports_dir: Path = typer.Option(
        REPORTS_DIR, help="Directory to save evaluation reports"
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Directory to save evaluation figures"
    ),
    test_episodes: int = typer.Option(
        15, help="Number of episodes to run for evaluation"
    ),
    max_steps_per_episode: int = typer.Option(
        20000, help="Maximum steps per evaluation episode"
    ),
    save_predictions: bool = typer.Option(
        True, help="Save detailed predictions to CSV"
    ),
    use_best_model: bool = typer.Option(
        True, help="Use best model instead of final model"
    ),
    detailed_analysis: bool = typer.Option(
        True, help="Perform detailed error analysis"
    ),
    confidence_threshold: float = typer.Option(
        0.8, help="Confidence threshold for high-confidence predictions"
    ),
):
    """Enhanced evaluation of trained DQN agent on IDS detection test set."""
    start_time = time.time()
    logger.info("🧪 Starting Enhanced DQN Agent Evaluation")
    logger.info("=" * 70)

    # Check if test data exists
    if not test_data_path.exists():
        logger.error(f"❌ Test data not found: {test_data_path}")
        logger.info(
            "💡 Please run 'python -m rl_ids.make_dataset' first to generate processed data")
        raise typer.Exit(1)

    # Determine which model to use
    if use_best_model:
        best_model_path = MODELS_DIR / "dqn_model_best.pt"
        if best_model_path.exists():
            model_path = best_model_path
            logger.info("🏆 Using best validation model")
        else:
            logger.warning("⚠️  Best model not found, using final model")
            use_best_model = False

    logger.info(f"📂 Using model: {model_path}")

    # Create output directories
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load test data
    logger.info(f"📂 Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # Load training data for label mapping reference
    train_df = None
    if train_data_path.exists():
        logger.info("📂 Loading training data for label mapping")
        train_df = pd.read_csv(train_data_path)

    # Get feature columns (exclude label columns)
    feature_cols = [col for col in test_df.columns if col not in [
        "Label", "Label_Original"]]

    # Get dimensions
    input_dim = len(feature_cols)
    n_classes = len(np.unique(test_df["Label"].values))

    logger.info(f"📊 Test dataset shape: {test_df.shape}")
    logger.info(f"🔢 Input dimension: {input_dim}")
    logger.info(f"🏷️  Number of classes: {n_classes}")

    # Log test class distribution
    logger.info("📈 Test class distribution:")
    test_dist = test_df['Label'].value_counts().sort_index()
    class_balance_info = {}
    for label, count in test_dist.items():
        percentage = count / len(test_df) * 100
        class_balance_info[label] = {'count': count, 'percentage': percentage}
        logger.info(
            f"   Class {label}: {count:8,} samples ({percentage:5.1f}%)")

    # Create label mapping for better interpretability
    label_mapping = {}
    if 'Label_Original' in test_df.columns:
        label_mapping = test_df.set_index(
            'Label')['Label_Original'].drop_duplicates().to_dict()
        logger.info(
            f"🏷️  Label mapping available: {len(label_mapping)} classes")

    # Initialize environment
    logger.info("🌍 Initializing test environment...")
    env = IDSDetectionEnv(
        data_path=test_data_path,
        feature_cols=feature_cols,
        label_col="Label"
    )

    # Load trained agent
    logger.info(f"🤖 Loading trained model from {model_path}")

    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        raise typer.Exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    if device.type == "cuda":
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config_dict = checkpoint["config"]
        training_info = checkpoint.get("training_info", {})

        if training_info:
            logger.info(
                f"📊 Model trained for {training_info.get('episode', 'unknown')} episodes")
            if 'best_accuracy' in training_info:
                logger.info(
                    f"🎯 Best training accuracy: {training_info['best_accuracy']:.4f}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise typer.Exit(1)

    # Create agent configuration
    config = DQNConfig(**config_dict)
    agent = DQNAgent(config=config)

    # Load model state
    agent.load_model(model_path)
    agent.model.to(device)
    agent.target_model.to(device)
    agent.epsilon = 0.0  # Pure greedy for evaluation

    logger.info("✅ Model loaded successfully. Starting evaluation...")

    # Enhanced evaluation loop
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    episode_accuracies = []
    episode_rewards = []
    episode_details = []
    prediction_times = []

    logger.info(f"🧪 Running {test_episodes} evaluation episodes...")

    for episode in tqdm(range(test_episodes), desc="Evaluation Episodes"):
        episode_start_time = time.time()

        state, info = env.reset()
        done = False
        episode_predictions = []
        episode_true_labels = []
        episode_confidences = []
        total_reward = 0
        step_count = 0
        episode_prediction_times = []

        while not done and step_count < max_steps_per_episode:
            step_count += 1

            # Measure prediction time
            pred_start_time = time.time()

            # Agent selects action (greedy only) with confidence
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.model(state_tensor)
                action_probs = torch.softmax(q_values, dim=1)
                action = torch.argmax(q_values, dim=1).item()
                confidence = action_probs[0][action].item()

            pred_time = time.time() - pred_start_time
            episode_prediction_times.append(pred_time)

            # Environment step
            next_state, reward, done, truncated, info = env.step(action=action)

            # Store predictions and true labels
            actual_label = info.get("actual_label", -1)
            if actual_label != -1:
                episode_predictions.append(action)
                episode_true_labels.append(actual_label)
                episode_confidences.append(confidence)
                all_predictions.append(action)
                all_true_labels.append(actual_label)
                all_confidences.append(confidence)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        # Calculate episode metrics
        if episode_true_labels:
            episode_accuracy = accuracy_score(
                episode_true_labels, episode_predictions)
            episode_accuracies.append(episode_accuracy)
            episode_rewards.append(total_reward)

            avg_confidence = np.mean(episode_confidences)
            avg_pred_time = np.mean(episode_prediction_times)
            prediction_times.extend(episode_prediction_times)

            episode_details.append({
                'episode': episode + 1,
                'accuracy': episode_accuracy,
                'reward': total_reward,
                'steps': step_count,
                'predictions': len(episode_predictions),
                'avg_confidence': avg_confidence,
                'avg_prediction_time_ms': avg_pred_time * 1000,
                'duration_seconds': time.time() - episode_start_time
            })

            logger.debug(
                f"Episode {episode + 1:2d}/{test_episodes}: "
                f"Acc: {episode_accuracy:.4f}, "
                f"Reward: {total_reward:7.2f}, "
                f"Conf: {avg_confidence:.3f}, "
                f"Time: {time.time() - episode_start_time:.1f}s"
            )

    # Calculate comprehensive metrics
    if not all_true_labels:
        logger.error("❌ No predictions collected during evaluation")
        raise typer.Exit(1)

    # Basic metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    avg_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    std_accuracy = np.std(episode_accuracies) if episode_accuracies else 0
    avg_confidence = np.mean(all_confidences)
    avg_prediction_time = np.mean(prediction_times) * 1000  # in ms

    # High-confidence predictions analysis
    high_conf_mask = np.array(all_confidences) >= confidence_threshold
    high_conf_accuracy = accuracy_score(
        np.array(all_true_labels)[high_conf_mask],
        np.array(all_predictions)[high_conf_mask]
    ) if np.sum(high_conf_mask) > 0 else 0.0

    high_conf_percentage = (np.sum(high_conf_mask) /
                            len(all_confidences)) * 100

    # Performance stability metrics
    # Coefficient of variation
    accuracy_cv = std_accuracy / (avg_accuracy + 1e-8)
    reward_cv = std_reward / (abs(avg_reward) + 1e-8)

    logger.info("\n" + "📊 ENHANCED EVALUATION RESULTS" + "\n" + "=" * 70)
    logger.info(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(
        f"📈 Average Episode Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(
        f"🏆 High-Confidence Accuracy (≥{confidence_threshold:.1f}): {high_conf_accuracy:.4f}")
    logger.info(f"💪 High-Confidence Predictions: {high_conf_percentage:.1f}%")
    logger.info(
        f"🎁 Average Reward per Episode: {avg_reward:.2f} (±{std_reward:.2f})")
    logger.info(f"📊 Total Predictions: {len(all_predictions):,}")
    logger.info(f"⚡ Average Prediction Time: {avg_prediction_time:.2f}ms")
    logger.info(f"📊 Accuracy Stability (CV): {accuracy_cv:.4f}")
    logger.info(f"🎲 Test Episodes: {test_episodes}")

    # Get unique labels that actually appear in predictions
    unique_labels = sorted(list(set(all_true_labels + all_predictions)))
    logger.info(f"🔢 Unique labels encountered: {unique_labels}")

    # Create class names
    if label_mapping:
        class_names = [
            f"{label_mapping.get(i, f'Class_{i}')}" for i in unique_labels]
    else:
        class_names = [f"Class_{i}" for i in unique_labels]

    # Generate enhanced classification report
    logger.info("📋 Generating enhanced classification report...")

    try:
        report_dict = classification_report(
            all_true_labels,
            all_predictions,
            labels=unique_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Print classification report
        print("\n" + "=" * 70)
        print("ENHANCED CLASSIFICATION REPORT")
        print("=" * 70)
        print(
            classification_report(
                all_true_labels,
                all_predictions,
                labels=unique_labels,
                target_names=class_names,
                zero_division=0,
            )
        )

        # Save classification report
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = reports_dir / "evaluation_classification_report.csv"
        report_df.to_csv(report_path)
        logger.success(f"📋 Classification report saved to: {report_path}")

    except Exception as e:
        logger.error(f"❌ Failed to generate classification report: {str(e)}")
        report_dict = {}

    # Enhanced confusion matrix analysis
    logger.info("🔢 Generating enhanced confusion matrix...")
    cm = confusion_matrix(
        all_true_labels, all_predictions, labels=unique_labels)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, all_predictions, labels=unique_labels, zero_division=0
    )

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = reports_dir / "evaluation_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.success(f"🔢 Confusion matrix saved to: {cm_path}")

    # Enhanced confusion matrix plot
    plt.figure(figsize=(max(12, len(unique_labels) * 1.2),
               max(10, len(unique_labels) * 1.0)))

    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create subplot for normalized confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax1
    )
    ax1.set_title("Confusion Matrix - Raw Counts",
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)

    # Normalized percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'},
        ax=ax2
    )
    ax2.set_title("Confusion Matrix - Normalized",
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("Predicted Label", fontsize=12)
    ax2.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    cm_plot_path = figures_dir / "evaluation_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(
        f"📊 Enhanced confusion matrix plot saved to: {cm_plot_path}")

    # Save enhanced episode details
    episodes_df = pd.DataFrame(episode_details)
    episodes_path = reports_dir / "evaluation_episode_details.csv"
    episodes_df.to_csv(episodes_path, index=False)
    logger.success(f"📝 Enhanced episode details saved to: {episodes_path}")

    # Enhanced per-class analysis
    if detailed_analysis and report_dict and len(unique_labels) > 1:
        logger.info("🔍 Performing detailed error analysis...")

        # Class-wise confidence analysis
        class_confidence_stats = {}
        class_error_analysis = {}

        for i, label in enumerate(unique_labels):
            label_mask = np.array(all_true_labels) == label
            if np.sum(label_mask) > 0:
                class_confidences = np.array(all_confidences)[label_mask]
                class_predictions = np.array(all_predictions)[label_mask]
                correct_mask = class_predictions == label

                class_confidence_stats[label] = {
                    'avg_confidence': np.mean(class_confidences),
                    'avg_confidence_correct': np.mean(class_confidences[correct_mask]) if np.sum(correct_mask) > 0 else 0,
                    'avg_confidence_incorrect': np.mean(class_confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0,
                    'high_conf_ratio': np.sum(class_confidences >= confidence_threshold) / len(class_confidences)
                }

                # Error analysis
                if np.sum(~correct_mask) > 0:
                    incorrect_preds = class_predictions[~correct_mask]
                    error_dist = {}
                    for pred in incorrect_preds:
                        error_dist[pred] = error_dist.get(pred, 0) + 1
                    class_error_analysis[label] = error_dist

        # Create enhanced visualization
        create_enhanced_visualizations(
            unique_labels, class_names, report_dict, class_confidence_stats,
            class_error_analysis, episodes_df, figures_dir, confidence_threshold
        )

    # Save detailed predictions with confidence scores
    if save_predictions:
        logger.info("💾 Saving detailed predictions with confidence scores...")
        predictions_df = pd.DataFrame({
            "True_Label": all_true_labels,
            "Predicted_Label": all_predictions,
            "Confidence": all_confidences,
            "Correct": np.array(all_true_labels) == np.array(all_predictions),
            "High_Confidence": np.array(all_confidences) >= confidence_threshold,
        })

        # Add original class names if available
        if label_mapping:
            predictions_df["True_Class"] = [label_mapping.get(
                label, f"Class_{label}") for label in all_true_labels]
            predictions_df["Predicted_Class"] = [label_mapping.get(
                label, f"Class_{label}") for label in all_predictions]

        predictions_path = reports_dir / "evaluation_detailed_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.success(
            f"📝 Enhanced detailed predictions saved to: {predictions_path}")

    # Enhanced evaluation summary
    total_eval_time = time.time() - start_time

    summary_dict = {
        "overall_accuracy": overall_accuracy,
        "average_episode_accuracy": avg_accuracy,
        "std_episode_accuracy": std_accuracy,
        "accuracy_cv": accuracy_cv,
        "high_confidence_accuracy": high_conf_accuracy,
        "high_confidence_percentage": high_conf_percentage,
        "average_reward_per_episode": avg_reward,
        "std_reward_per_episode": std_reward,
        "reward_cv": reward_cv,
        "average_confidence": avg_confidence,
        "avg_prediction_time_ms": avg_prediction_time,
        "total_predictions": len(all_predictions),
        "number_of_classes": len(unique_labels),
        "test_episodes": test_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "confidence_threshold": confidence_threshold,
        "total_evaluation_time_minutes": total_eval_time / 60,
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
        "used_best_model": use_best_model,
    }

    # Add macro and weighted averages if available
    if report_dict:
        if "macro avg" in report_dict:
            summary_dict.update({
                "macro_avg_precision": report_dict["macro avg"]["precision"],
                "macro_avg_recall": report_dict["macro avg"]["recall"],
                "macro_avg_f1_score": report_dict["macro avg"]["f1-score"],
            })

        if "weighted avg" in report_dict:
            summary_dict.update({
                "weighted_avg_precision": report_dict["weighted avg"]["precision"],
                "weighted_avg_recall": report_dict["weighted avg"]["recall"],
                "weighted_avg_f1_score": report_dict["weighted avg"]["f1-score"],
            })

    summary_df = pd.DataFrame([summary_dict])
    summary_path = reports_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.success(f"📊 Enhanced evaluation summary saved to: {summary_path}")

    # Final enhanced summary
    logger.info("\n" + "✅ ENHANCED EVALUATION COMPLETED!" + "\n" + "=" * 70)
    logger.info(
        f"⏱️  Total evaluation time: {total_eval_time / 60:.1f} minutes")
    logger.info(f"📁 Reports saved to: {reports_dir}")
    logger.info(f"📈 Figures saved to: {figures_dir}")
    logger.info(
        f"🏆 Model type: {'Best Validation' if use_best_model else 'Final Training'}")

    print("\n" + "=" * 70)
    print("ENHANCED EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(
        f"High-Confidence Accuracy: {high_conf_accuracy:.4f} ({high_conf_percentage:.1f}% of predictions)")
    print(
        f"Average Episode Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Performance Stability (CV): {accuracy_cv:.4f}")
    if report_dict and "macro avg" in report_dict:
        print(
            f"Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
        print(f"Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
        print(
            f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Average Prediction Time: {avg_prediction_time:.2f}ms")
    print(f"Total Predictions: {len(all_predictions):,}")
    print(f"Test Episodes: {test_episodes}")
    print("=" * 70)

    return {
        "overall_accuracy": overall_accuracy,
        "high_confidence_accuracy": high_conf_accuracy,
        "average_reward": avg_reward,
        "average_confidence": avg_confidence,
        "total_predictions": len(all_predictions),
        "evaluation_time_minutes": total_eval_time / 60,
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir),
    }


def create_enhanced_visualizations(unique_labels, class_names, report_dict,
                                   class_confidence_stats, class_error_analysis,
                                   episodes_df, figures_dir, confidence_threshold):
    """Create enhanced visualizations for detailed analysis"""

    # 1. Performance metrics over episodes
    plt.figure(figsize=(15, 10))

    # Episode accuracy and confidence trends
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    episodes = episodes_df['episode']

    # Accuracy trend
    ax1.plot(episodes, episodes_df['accuracy'], 'b-', linewidth=2, alpha=0.7)
    ax1.axhline(y=episodes_df['accuracy'].mean(), color='r', linestyle='--',
                label=f'Mean: {episodes_df["accuracy"].mean():.3f}')
    ax1.set_title('Accuracy per Episode', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confidence trend
    ax2.plot(episodes, episodes_df['avg_confidence'],
             'g-', linewidth=2, alpha=0.7)
    ax2.axhline(y=confidence_threshold, color='r', linestyle='--',
                label=f'Threshold: {confidence_threshold}')
    ax2.set_title('Average Confidence per Episode', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Reward trend
    ax3.plot(episodes, episodes_df['reward'], 'purple', linewidth=2, alpha=0.7)
    ax3.axhline(y=episodes_df['reward'].mean(), color='r', linestyle='--',
                label=f'Mean: {episodes_df["reward"].mean():.1f}')
    ax3.set_title('Reward per Episode', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Prediction time trend
    ax4.plot(episodes, episodes_df['avg_prediction_time_ms'],
             'orange', linewidth=2, alpha=0.7)
    ax4.axhline(y=episodes_df['avg_prediction_time_ms'].mean(), color='r', linestyle='--',
                label=f'Mean: {episodes_df["avg_prediction_time_ms"].mean():.2f}ms')
    ax4.set_title('Average Prediction Time per Episode', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Time (ms)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    trends_path = figures_dir / "evaluation_episode_trends.png"
    plt.savefig(trends_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Class-wise confidence analysis
    if class_confidence_stats:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        labels = list(class_confidence_stats.keys())
        class_names_short = [class_names[i] for i in range(len(labels))]

        # Average confidence per class
        avg_confidences = [class_confidence_stats[l]
                           ['avg_confidence'] for l in labels]
        bars1 = ax1.bar(range(len(labels)), avg_confidences,
                        color='skyblue', alpha=0.7)
        ax1.set_title('Average Confidence per Class')
        ax1.set_ylabel('Confidence')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(class_names_short, rotation=45, ha='right')
        ax1.axhline(y=confidence_threshold, color='r',
                    linestyle='--', alpha=0.7)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{avg_confidences[i]:.3f}', ha='center', va='bottom')

        # High confidence ratio per class
        high_conf_ratios = [class_confidence_stats[l]
                            ['high_conf_ratio'] for l in labels]
        bars2 = ax2.bar(range(len(labels)), high_conf_ratios,
                        color='lightgreen', alpha=0.7)
        ax2.set_title(
            f'High Confidence Ratio per Class (≥{confidence_threshold})')
        ax2.set_ylabel('Ratio')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(class_names_short, rotation=45, ha='right')
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{high_conf_ratios[i]:.3f}', ha='center', va='bottom')

        # Confidence difference (correct vs incorrect)
        conf_correct = [class_confidence_stats[l]
                        ['avg_confidence_correct'] for l in labels]
        conf_incorrect = [class_confidence_stats[l]
                          ['avg_confidence_incorrect'] for l in labels]

        x = np.arange(len(labels))
        width = 0.35
        ax3.bar(x - width/2, conf_correct, width,
                label='Correct', alpha=0.8, color='green')
        ax3.bar(x + width/2, conf_incorrect, width,
                label='Incorrect', alpha=0.8, color='red')
        ax3.set_title('Confidence: Correct vs Incorrect Predictions')
        ax3.set_ylabel('Confidence')
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names_short, rotation=45, ha='right')
        ax3.legend()

        # Performance metrics per class
        if report_dict:
            precisions = []
            recalls = []
            f1_scores = []

            for i, label in enumerate(labels):
                class_key = class_names_short[i]
                if class_key in report_dict:
                    precisions.append(
                        report_dict[class_key].get('precision', 0.0))
                    recalls.append(report_dict[class_key].get('recall', 0.0))
                    f1_scores.append(
                        report_dict[class_key].get('f1-score', 0.0))
                else:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    f1_scores.append(0.0)

            x = np.arange(len(labels))
            width = 0.25
            ax4.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax4.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax4.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            ax4.set_title('Performance Metrics per Class')
            ax4.set_ylabel('Score')
            ax4.set_xticks(x)
            ax4.set_xticklabels(class_names_short, rotation=45, ha='right')
            ax4.legend()
            ax4.set_ylim(0, 1)

        plt.tight_layout()
        confidence_analysis_path = figures_dir / "evaluation_confidence_analysis.png"
        plt.savefig(confidence_analysis_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    app()

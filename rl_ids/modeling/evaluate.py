import os
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from tqdm import tqdm
import typer

from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from rl_ids.environments.ids_env import IDSDetectionEnv

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "dqn_model_final.pt", help="Path to trained DQN model"
    ),
    test_data_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test.csv", help="Path to test dataset"
    ),
    train_data_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "train.csv", help="Path to training dataset (for label mapping)"
    ),
    reports_dir: Path = typer.Option(
        REPORTS_DIR, help="Directory to save evaluation reports"
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Directory to save evaluation figures"
    ),
    test_episodes: int = typer.Option(
        10, help="Number of episodes to run for evaluation"
    ),
    max_steps_per_episode: int = typer.Option(
        20000, help="Maximum steps per evaluation episode"
    ),
    save_predictions: bool = typer.Option(
        True, help="Save detailed predictions to CSV"
    ),
    use_best_model: bool = typer.Option(
        False, help="Use best model instead of final model"
    ),
):
    """Evaluate trained DQN agent on IDS detection test set."""
    logger.info("🧪 Starting DQN Agent Evaluation")
    logger.info("=" * 60)

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
        logger.info(f"📂 Loading training data for label mapping")
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
    for label, count in test_dist.items():
        percentage = count / len(test_df) * 100
        logger.info(
            f"   Class {label}: {count:8,} samples ({percentage:5.1f}%)")

    # Create label mapping for better interpretability
    label_mapping = {}
    if 'Label_Original' in test_df.columns:
        label_mapping = test_df.set_index(
            'Label')['Label_Original'].drop_duplicates().to_dict()
        logger.info(f"🏷️  Label mapping: {label_mapping}")

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

    # Evaluation loop
    all_predictions = []
    all_true_labels = []
    episode_accuracies = []
    episode_rewards = []
    episode_details = []

    logger.info(f"🧪 Running {test_episodes} evaluation episodes...")

    for episode in tqdm(range(test_episodes), desc="Evaluation Episodes"):
        state, info = env.reset()
        done = False
        episode_predictions = []
        episode_true_labels = []
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            step_count += 1

            # Agent selects action (greedy only)
            action = agent.act(state=state, training=False)

            # Environment step
            next_state, reward, done, truncated, info = env.step(action=action)

            # Store predictions and true labels
            actual_label = info.get("actual_label", -1)
            if actual_label != -1:
                episode_predictions.append(action)
                episode_true_labels.append(actual_label)
                all_predictions.append(action)
                all_true_labels.append(actual_label)

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

            episode_details.append({
                'episode': episode + 1,
                'accuracy': episode_accuracy,
                'reward': total_reward,
                'steps': step_count,
                'predictions': len(episode_predictions)
            })

            logger.debug(
                f"Episode {episode + 1:2d}/{test_episodes}: "
                f"Acc: {episode_accuracy:.4f}, "
                f"Reward: {total_reward:7.2f}, "
                f"Steps: {step_count:5d}, "
                f"Predictions: {len(episode_predictions):5d}"
            )

    # Calculate overall metrics
    if not all_true_labels:
        logger.error("❌ No predictions collected during evaluation")
        raise typer.Exit(1)

    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    avg_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    std_accuracy = np.std(episode_accuracies) if episode_accuracies else 0

    logger.info("\n" + "📊 EVALUATION RESULTS" + "\n" + "=" * 50)
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(
        f"Average Episode Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(
        f"Average Reward per Episode: {avg_reward:.2f} (±{std_reward:.2f})")
    logger.info(f"Total Predictions: {len(all_predictions):,}")
    logger.info(f"Test Episodes: {test_episodes}")

    # Get unique labels that actually appear in predictions
    unique_labels = sorted(list(set(all_true_labels + all_predictions)))
    logger.info(f"Unique labels encountered: {unique_labels}")

    # Create class names
    if label_mapping:
        class_names = [
            f"{label_mapping.get(i, f'Class_{i}')}" for i in unique_labels]
    else:
        class_names = [f"Class_{i}" for i in unique_labels]

    # Generate classification report
    logger.info("📋 Generating classification report...")

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
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
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

    # Generate and save confusion matrix
    logger.info("🔢 Generating confusion matrix...")
    cm = confusion_matrix(
        all_true_labels, all_predictions, labels=unique_labels)

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = reports_dir / "evaluation_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.success(f"🔢 Confusion matrix saved to: {cm_path}")

    # Plot confusion matrix
    plt.figure(figsize=(max(10, len(unique_labels) * 0.8),
               max(8, len(unique_labels) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title("Confusion Matrix - DQN Agent Evaluation",
              fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_plot_path = figures_dir / "evaluation_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"📊 Confusion matrix plot saved to: {cm_plot_path}")

    # Save episode details
    episodes_df = pd.DataFrame(episode_details)
    episodes_path = reports_dir / "evaluation_episode_details.csv"
    episodes_df.to_csv(episodes_path, index=False)
    logger.success(f"📝 Episode details saved to: {episodes_path}")

    # Plot per-class metrics if we have classification report
    if report_dict and len(unique_labels) > 1:
        logger.info("📈 Creating per-class metrics plots...")

        # Extract per-class metrics
        class_metrics = {}
        for i, label in enumerate(unique_labels):
            class_key = class_names[i]
            if class_key in report_dict:
                class_metrics[label] = {
                    'precision': report_dict[class_key].get('precision', 0.0),
                    'recall': report_dict[class_key].get('recall', 0.0),
                    'f1-score': report_dict[class_key].get('f1-score', 0.0),
                    'support': report_dict[class_key].get('support', 0)
                }

        if class_metrics:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(15, 12))

            labels = list(class_metrics.keys())
            precisions = [class_metrics[l]['precision'] for l in labels]
            recalls = [class_metrics[l]['recall'] for l in labels]
            f1_scores = [class_metrics[l]['f1-score'] for l in labels]
            supports = [class_metrics[l]['support'] for l in labels]

            # Precision
            bars1 = ax1.bar(range(len(labels)), precisions,
                            color='skyblue', alpha=0.7)
            ax1.set_title('Precision per Class')
            ax1.set_ylabel('Precision')
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels([class_names[i] for i in range(
                len(labels))], rotation=45, ha='right')
            ax1.set_ylim(0, 1)
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{precisions[i]:.3f}', ha='center', va='bottom')

            # Recall
            bars2 = ax2.bar(range(len(labels)), recalls,
                            color='lightgreen', alpha=0.7)
            ax2.set_title('Recall per Class')
            ax2.set_ylabel('Recall')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels([class_names[i] for i in range(
                len(labels))], rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{recalls[i]:.3f}', ha='center', va='bottom')

            # F1-Score
            bars3 = ax3.bar(range(len(labels)), f1_scores,
                            color='lightcoral', alpha=0.7)
            ax3.set_title('F1-Score per Class')
            ax3.set_ylabel('F1-Score')
            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels([class_names[i] for i in range(
                len(labels))], rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{f1_scores[i]:.3f}', ha='center', va='bottom')

            # Support
            bars4 = ax4.bar(range(len(labels)), supports,
                            color='gold', alpha=0.7)
            ax4.set_title('Support per Class')
            ax4.set_ylabel('Sample Count')
            ax4.set_xticks(range(len(labels)))
            ax4.set_xticklabels([class_names[i] for i in range(
                len(labels))], rotation=45, ha='right')
            for i, bar in enumerate(bars4):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(supports) * 0.01,
                         f'{supports[i]}', ha='center', va='bottom')

            plt.tight_layout()
            metrics_plot_path = figures_dir / "evaluation_per_class_metrics.png"
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.success(
                f"📈 Per-class metrics plot saved to: {metrics_plot_path}")

    # Save detailed predictions if requested
    if save_predictions:
        logger.info("💾 Saving detailed predictions...")
        predictions_df = pd.DataFrame({
            "True_Label": all_true_labels,
            "Predicted_Label": all_predictions,
            "Correct": np.array(all_true_labels) == np.array(all_predictions),
        })

        # Add original class names if available
        if label_mapping:
            predictions_df["True_Class"] = [label_mapping.get(
                label, f"Class_{label}") for label in all_true_labels]
            predictions_df["Predicted_Class"] = [label_mapping.get(
                label, f"Class_{label}") for label in all_predictions]

        predictions_path = reports_dir / "evaluation_detailed_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.success(f"📝 Detailed predictions saved to: {predictions_path}")

    # Save evaluation summary
    summary_dict = {
        "overall_accuracy": overall_accuracy,
        "average_episode_accuracy": avg_accuracy,
        "std_episode_accuracy": std_accuracy,
        "average_reward_per_episode": avg_reward,
        "std_reward_per_episode": std_reward,
        "total_predictions": len(all_predictions),
        "number_of_classes": len(unique_labels),
        "test_episodes": test_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
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
    logger.success(f"📊 Evaluation summary saved to: {summary_path}")

    # Create overall performance summary plot
    if report_dict and "macro avg" in report_dict and "weighted avg" in report_dict:
        plt.figure(figsize=(12, 8))

        metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]
        macro_scores = [
            report_dict["macro avg"]["precision"],
            report_dict["macro avg"]["recall"],
            report_dict["macro avg"]["f1-score"],
            overall_accuracy
        ]
        weighted_scores = [
            report_dict["weighted avg"]["precision"],
            report_dict["weighted avg"]["recall"],
            report_dict["weighted avg"]["f1-score"],
            overall_accuracy
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = plt.bar(x - width/2, macro_scores, width,
                        label="Macro Average", alpha=0.8, color='steelblue')
        bars2 = plt.bar(x + width/2, weighted_scores, width,
                        label="Weighted Average", alpha=0.8, color='darkorange')

        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Overall Performance Metrics - DQN IDS Evaluation",
                  fontsize=14, fontweight='bold')
        plt.xticks(x, metrics)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (macro, weighted) in enumerate(zip(macro_scores, weighted_scores)):
            plt.text(i - width/2, macro + 0.01,
                     f"{macro:.3f}", ha="center", va="bottom")
            plt.text(i + width/2, weighted + 0.01,
                     f"{weighted:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        performance_plot_path = figures_dir / "evaluation_performance_summary.png"
        plt.savefig(performance_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.success(
            f"📊 Performance summary plot saved to: {performance_plot_path}")

    logger.info("\n" + "✅ EVALUATION COMPLETED SUCCESSFULLY!" + "\n" + "=" * 60)
    logger.info(f"📁 Reports saved to: {reports_dir}")
    logger.info(f"📈 Figures saved to: {figures_dir}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(
        f"Average Episode Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    if report_dict and "macro avg" in report_dict:
        print(
            f"Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
        print(f"Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
        print(
            f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Total Predictions: {len(all_predictions):,}")
    print(f"Test Episodes: {test_episodes}")
    print("=" * 60)

    return {
        "overall_accuracy": overall_accuracy,
        "average_reward": avg_reward,
        "total_predictions": len(all_predictions),
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir),
    }


if __name__ == "__main__":
    app()

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm

from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.config import NORMALISED_DATA_FILE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "dqn_model_final.pt",
        help="Path to trained DQN model"
    ),
    data_path: Path = typer.Option(
        NORMALISED_DATA_FILE,
        help="Path to evaluation dataset"
    ),
    reports_dir: Path = typer.Option(
        REPORTS_DIR,
        help="Directory to save evaluation reports"
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR,
        help="Directory to save evaluation figures"
    ),
    test_episodes: int = typer.Option(
        10,
        help="Number of episodes to run for evaluation"
    ),
    max_steps_per_episode: int = typer.Option(
        20000,
        help="Maximum steps per evaluation episode"
    ),
    save_predictions: bool = typer.Option(
        True,
        help="Save detailed predictions to CSV"
    )
):
    """Evaluate trained DQN agent on IDS detection task."""

    logger.info("Starting DQN Agent Evaluation")

    # Create output directories
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load and prepare data
    logger.info(f"Loading evaluation data from {data_path}")
    df = pd.read_csv(data_path)

    # Get feature columns (exclude label columns)
    feature_cols = [col for col in df.columns if col not in [
        'Label', 'Label_Original']]

    # Get dimensions
    input_dim = len(feature_cols)
    n_classes = len(np.unique(df['Label'].values))

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class distribution:\n{df['Label'].value_counts()}")

    # Initialize environment
    logger.info("Initializing evaluation environment...")
    env = IDSDetectionEnv(
        data_path=data_path,
        feature_cols=feature_cols,
        label_col="Label"
    )

    # Load trained agent
    logger.info(f"Loading trained model from {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise typer.Exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']

    # Create agent configuration
    config = DQNConfig(**config_dict)
    agent = DQNAgent(config=config)

    # Load model state
    agent.load_model(model_path)
    agent.model.to(device)  # Ensure model is on correct device
    agent.target_model.to(device)  # Ensure target model is on correct device
    agent.epsilon = 0.0  # Pure greedy for evaluation

    logger.info("Model loaded successfully. Starting evaluation...")

    # Evaluation loop
    all_predictions = []
    all_true_labels = []
    episode_accuracies = []
    episode_rewards = []

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

            logger.info(
                f"Episode {episode + 1}/{test_episodes}: "
                f"Accuracy: {episode_accuracy:.4f}, "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {step_count}"
            )

    # Calculate overall metrics
    if not all_true_labels:
        logger.error("No predictions collected during evaluation")
        raise typer.Exit(1)

    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0

    logger.info(f"\n📊 Evaluation Results:")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Average Reward per Episode: {avg_reward:.2f}")
    logger.info(f"Total Predictions: {len(all_predictions)}")

    # Generate classification report
    logger.info("Generating classification report...")

    # Get unique labels that actually appear in predictions
    unique_labels = sorted(list(set(all_true_labels + all_predictions)))
    n_unique_labels = len(unique_labels)

    logger.info(f"Unique labels encountered: {unique_labels}")
    logger.info(f"Number of unique labels: {n_unique_labels}")

    # Create class names only for labels that appear
    class_names = [f"Class_{i}" for i in unique_labels]

    # Classification report as dictionary
    report_dict = classification_report(
        all_true_labels,
        all_predictions,
        labels=unique_labels,  # Specify which labels to include
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(
        all_true_labels,
        all_predictions,
        labels=unique_labels,  # Specify which labels to include
        target_names=class_names,
        zero_division=0
    ))

    # Save classification report
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = reports_dir / "evaluation_classification_report.csv"
    report_df.to_csv(report_path)
    logger.success(f"Classification report saved to: {report_path}")

    # Generate and save confusion matrix
    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(
        all_true_labels, all_predictions, labels=unique_labels)

    # Save confusion matrix as CSV
    cm_path = reports_dir / "evaluation_confusion_matrix.csv"
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
    logger.success(f"Confusion matrix saved to: {cm_path}")

    # Plot confusion matrix
    plt.figure(figsize=(max(8, len(unique_labels)), max(6, len(unique_labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix - DQN Agent Evaluation")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_plot_path = figures_dir / "evaluation_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Confusion matrix plot saved to: {cm_plot_path}")

    # Plot accuracy per class
    if len(set(all_true_labels)) > 1:
        plt.figure(figsize=(12, 6))
        class_accuracies = []
        class_counts = []

        # Use unique_labels instead of range(n_classes)
        for class_id in unique_labels:
            class_mask = np.array(all_true_labels) == class_id
            class_count = np.sum(class_mask)
            class_counts.append(class_count)

            if class_count > 0:
                class_acc = accuracy_score(
                    np.array(all_true_labels)[class_mask],
                    np.array(all_predictions)[class_mask]
                )
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)

        # Create subplot for accuracy and counts
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot accuracy per class
        bars1 = ax1.bar(range(len(unique_labels)), class_accuracies,
                        color='skyblue', alpha=0.7)
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Per-Class Accuracy")
        ax1.set_xticks(range(len(unique_labels)))
        ax1.set_xticklabels(class_names, rotation=45)
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{class_accuracies[i]:.3f}', ha='center', va='bottom')

        # Plot sample counts per class
        bars2 = ax2.bar(range(len(unique_labels)), class_counts,
                        color='lightcoral', alpha=0.7)
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Sample Count")
        ax2.set_title("Sample Count per Class")
        ax2.set_xticks(range(len(unique_labels)))
        ax2.set_xticklabels(class_names, rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if max(class_counts) > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                         f'{class_counts[i]}', ha='center', va='bottom')

        plt.tight_layout()

        acc_plot_path = figures_dir / "evaluation_per_class_metrics.png"
        plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Per-class metrics plot saved to: {acc_plot_path}")
    else:
        logger.warning(
            "Only one class encountered in predictions. Skipping per-class accuracy plot.")

    # Save detailed predictions if requested
    if save_predictions:
        logger.info("Saving detailed predictions...")
        predictions_df = pd.DataFrame({
            "True_Label": all_true_labels,
            "Predicted_Label": all_predictions,
            "Correct": np.array(all_true_labels) == np.array(all_predictions)
        })

        # Add class names for better readability
        predictions_df["True_Class"] = [
            f"Class_{label}" for label in all_true_labels]
        predictions_df["Predicted_Class"] = [
            f"Class_{label}" for label in all_predictions]

        predictions_path = reports_dir / "evaluation_detailed_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.success(f"Detailed predictions saved to: {predictions_path}")

    # Save evaluation summary
    summary_dict = {
        "overall_accuracy": overall_accuracy,
        "average_reward_per_episode": avg_reward,
        "total_predictions": len(all_predictions),
        "number_of_classes": n_classes,
        "test_episodes": test_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "model_path": str(model_path),
        "data_path": str(data_path)
    }

    # Add macro and weighted averages
    if 'macro avg' in report_dict:
        summary_dict.update({
            "macro_avg_precision": report_dict['macro avg']['precision'],
            "macro_avg_recall": report_dict['macro avg']['recall'],
            "macro_avg_f1_score": report_dict['macro avg']['f1-score']
        })

    if 'weighted avg' in report_dict:
        summary_dict.update({
            "weighted_avg_precision": report_dict['weighted avg']['precision'],
            "weighted_avg_recall": report_dict['weighted avg']['recall'],
            "weighted_avg_f1_score": report_dict['weighted avg']['f1-score']
        })

    # Add per-class metrics (only for classes that actually appear)
    for class_id in unique_labels:
        class_key = f"Class_{class_id}"
        if class_key in report_dict:
            summary_dict[f"precision_class_{class_id}"] = report_dict[class_key].get(
                "precision", 0.0)
            summary_dict[f"recall_class_{class_id}"] = report_dict[class_key].get(
                "recall", 0.0)
            summary_dict[f"f1_score_class_{class_id}"] = report_dict[class_key].get(
                "f1-score", 0.0)
            summary_dict[f"support_class_{class_id}"] = report_dict[class_key].get(
                "support", 0)

    summary_df = pd.DataFrame([summary_dict])
    summary_path = reports_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.success(f"Evaluation summary saved to: {summary_path}")

    # Create a performance summary plot
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1-Score']
    macro_scores = [
        report_dict['macro avg']['precision'],
        report_dict['macro avg']['recall'],
        report_dict['macro avg']['f1-score']
    ]
    weighted_scores = [
        report_dict['weighted avg']['precision'],
        report_dict['weighted avg']['recall'],
        report_dict['weighted avg']['f1-score']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, macro_scores, width, label='Macro Average', alpha=0.7)
    plt.bar(x + width/2, weighted_scores, width,
            label='Weighted Average', alpha=0.7)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Overall Performance Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)

    # Add value labels on bars
    for i, (macro, weighted) in enumerate(zip(macro_scores, weighted_scores)):
        plt.text(i - width/2, macro + 0.01,
                 f'{macro:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, weighted + 0.01,
                 f'{weighted:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    performance_plot_path = figures_dir / "evaluation_performance_summary.png"
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(
        f"Performance summary plot saved to: {performance_plot_path}")

    logger.info("\n✅ Evaluation completed successfully!")
    logger.info(f"📁 Reports saved to: {reports_dir}")
    logger.info(f"📈 Figures saved to: {figures_dir}")

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Total Predictions: {len(all_predictions):,}")
    print("="*50)

    return {
        "overall_accuracy": overall_accuracy,
        "average_reward": avg_reward,
        "total_predictions": len(all_predictions),
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir)
    }


if __name__ == "__main__":
    app()

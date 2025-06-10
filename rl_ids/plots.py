from pathlib import Path
from typing import Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from rl_ids.config import FIGURES_DIR, BALANCED_DATA_FILE, REPORTS_DIR

app = typer.Typer()


def plot_label_distribution(csv_path: Path, save_path: Path, label_col: str = "Label_Original"):
    """Plot class distribution from dataset."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        logger.warning(
            f"Column {label_col} not found. Available columns: {df.columns.tolist()}")
        # Try alternative label columns
        if "Label" in df.columns:
            label_col = "Label"
            logger.info(f"Using {label_col} instead")
        else:
            logger.error("No suitable label column found")
            return

    label_counts = df[label_col].value_counts()

    plt.figure(figsize=(14, 8))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution in Dataset", fontsize=16, fontweight="bold")
    plt.xlabel("Class Labels", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)

    # Add value labels on bars
    for i, v in enumerate(label_counts.values):
        plt.text(
            i,
            v + max(label_counts.values) * 0.01,
            str(v),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✅ Saved class distribution plot to {save_path}")


def plot_f1_per_class(report_path: Path, save_path: Path):
    """Plot F1-score per class from classification report."""
    logger.info(f"Loading classification report from {report_path}")

    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return

    report_df = pd.read_csv(report_path, index_col=0)

    # Remove summary rows
    exclude_rows = ["accuracy", "macro avg", "weighted avg"]
    for row in exclude_rows:
        if row in report_df.index:
            report_df = report_df.drop(row)

    if "f1-score" not in report_df.columns:
        logger.error("f1-score column not found in classification report")
        return

    plt.figure(figsize=(14, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(report_df)))
    bars = plt.bar(range(len(report_df)), report_df["f1-score"], color=colors)

    plt.xticks(range(len(report_df)), report_df.index, rotation=45, ha="right")
    plt.title("F1-Score per Class", fontsize=16, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.1)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, report_df["f1-score"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add horizontal line for average F1-score
    avg_f1 = report_df["f1-score"].mean()
    plt.axhline(
        y=avg_f1, color="red", linestyle="--", alpha=0.7, label=f"Average F1: {avg_f1:.3f}"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✅ Saved F1-score plot to {save_path}")


def plot_training_metrics(metrics_path: Path, save_path: Path):
    """Plot training metrics over episodes."""
    logger.info(f"Loading training metrics from {metrics_path}")

    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot rewards
    ax1.plot(df["Episode"], df["Reward"], color="blue", alpha=0.7)
    ax1.set_title("Reward per Episode", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    # Plot losses
    ax2.plot(df["Episode"], df["Loss"], color="red", alpha=0.7)
    ax2.set_title("Loss per Episode", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    # Plot accuracy
    ax3.plot(df["Episode"], df["Accuracy"], color="green", alpha=0.7)
    ax3.set_title("Accuracy per Episode", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot moving averages
    window = min(10, len(df) // 4)
    if len(df) > window:
        reward_ma = df["Reward"].rolling(window=window).mean()
        accuracy_ma = df["Accuracy"].rolling(window=window).mean()

        ax4.plot(df["Episode"], reward_ma,
                 label=f"Reward (MA-{window})", color="blue")
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df["Episode"], accuracy_ma,
                      label=f"Accuracy (MA-{window})", color="green")

        ax4.set_title("Moving Averages", fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Reward", color="blue")
        ax4_twin.set_ylabel("Accuracy", color="green")
        ax4.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✅ Saved training metrics plot to {save_path}")


def plot_confusion_matrix_heatmap(
    cm_path: Path, save_path: Path, class_names: Optional[list] = None
):
    """Plot confusion matrix as enhanced heatmap."""
    logger.info(f"Loading confusion matrix from {cm_path}")

    if not cm_path.exists():
        logger.error(f"Confusion matrix file not found: {cm_path}")
        return

    cm = np.loadtxt(cm_path, delimiter=",")

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
    )

    plt.title("Normalized Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✅ Saved enhanced confusion matrix plot to {save_path}")


@app.command()
def plot_data_distribution(
    data_path: Path = typer.Option(
        BALANCED_DATA_FILE, help="Path to dataset CSV"),
    output_path: Path = typer.Option(
        FIGURES_DIR / "class_distribution.png", help="Output plot path"
    ),
    label_col: str = typer.Option("Label_Original", help="Label column name"),
):
    """Generate class distribution plot from dataset."""
    plot_label_distribution(data_path, output_path, label_col)


@app.command()
def plot_classification_metrics(
    report_path: Path = typer.Option(
        REPORTS_DIR / "evaluation_classification_report.csv",
        help="Path to classification report CSV",
    ),
    output_path: Path = typer.Option(
        FIGURES_DIR / "f1_per_class.png", help="Output plot path"),
):
    """Generate F1-score per class plot from classification report."""
    plot_f1_per_class(report_path, output_path)


@app.command()
def plot_training_progress(
    metrics_path: Path = typer.Option(
        REPORTS_DIR / "training_metrics.csv", help="Path to training metrics CSV"
    ),
    output_path: Path = typer.Option(
        FIGURES_DIR / "training_progress.png", help="Output plot path"
    ),
):
    """Generate training progress plots from metrics."""
    plot_training_metrics(metrics_path, output_path)


@app.command()
def plot_enhanced_confusion_matrix(
    cm_path: Path = typer.Option(
        REPORTS_DIR / "evaluation_confusion_matrix.csv", help="Path to confusion matrix CSV"
    ),
    output_path: Path = typer.Option(
        FIGURES_DIR / "enhanced_confusion_matrix.png", help="Output plot path"
    ),
):
    """Generate enhanced confusion matrix plot."""
    plot_confusion_matrix_heatmap(cm_path, output_path)


@app.command()
def generate_all_plots(
    data_path: Path = typer.Option(
        BALANCED_DATA_FILE, help="Path to dataset CSV"),
    reports_dir: Path = typer.Option(REPORTS_DIR, help="Reports directory"),
    figures_dir: Path = typer.Option(
        FIGURES_DIR, help="Figures output directory"),
):
    """Generate all available plots."""
    logger.info("Generating all plots...")

    # Create figures directory
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    logger.info("1/5 - Generating class distribution plot...")
    plot_label_distribution(data_path, figures_dir / "class_distribution.png")

    # 2. F1-score per class (if evaluation report exists)
    eval_report = reports_dir / "evaluation_classification_report.csv"
    if eval_report.exists():
        logger.info("2/5 - Generating F1-score per class plot...")
        plot_f1_per_class(eval_report, figures_dir / "f1_per_class.png")
    else:
        logger.warning(
            "2/5 - Evaluation report not found, skipping F1-score plot")

    # 3. Training metrics (if training metrics exist)
    training_metrics = reports_dir / "training_metrics.csv"
    if training_metrics.exists():
        logger.info("3/5 - Generating training progress plot...")
        plot_training_metrics(
            training_metrics, figures_dir / "training_progress.png")
    else:
        logger.warning(
            "3/5 - Training metrics not found, skipping training progress plot")

    # 4. Enhanced confusion matrix (if confusion matrix exists)
    cm_file = reports_dir / "evaluation_confusion_matrix.csv"
    if cm_file.exists():
        logger.info("4/5 - Generating enhanced confusion matrix plot...")
        plot_confusion_matrix_heatmap(
            cm_file, figures_dir / "enhanced_confusion_matrix.png")
    else:
        logger.warning(
            "4/5 - Confusion matrix not found, skipping enhanced confusion matrix plot")

    # 5. Summary plot combining key metrics
    logger.info("5/5 - Generating summary dashboard...")
    create_summary_dashboard(data_path, reports_dir,
                             figures_dir / "summary_dashboard.png")

    logger.success("✅ All plots generated successfully!")


def create_summary_dashboard(data_path: Path, reports_dir: Path, save_path: Path):
    """Create a summary dashboard with key metrics."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Class distribution (simplified)
        df = pd.read_csv(data_path)
        label_col = "Label_Original" if "Label_Original" in df.columns else "Label"
        label_counts = df[label_col].value_counts()

        ax1.pie(
            label_counts.values[:10], labels=label_counts.index[:10], autopct="%1.1f%%")
        ax1.set_title("Top 10 Classes Distribution", fontweight="bold")

        # 2. Training accuracy over time (if available)
        training_metrics = reports_dir / "training_metrics.csv"
        if training_metrics.exists():
            metrics_df = pd.read_csv(training_metrics)
            ax2.plot(metrics_df["Episode"],
                     metrics_df["Accuracy"], color="green", linewidth=2)
            ax2.set_title("Training Accuracy Progress", fontweight="bold")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Accuracy")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "Training metrics\nnot available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title("Training Metrics", fontweight="bold")

        # 3. Model performance metrics (if available)
        eval_report = reports_dir / "evaluation_classification_report.csv"
        if eval_report.exists():
            report_df = pd.read_csv(eval_report, index_col=0)
            if "macro avg" in report_df.index:
                metrics = ["precision", "recall", "f1-score"]
                values = [report_df.loc["macro avg", metric]
                          for metric in metrics]

                bars = ax3.bar(metrics, values, color=[
                               "skyblue", "lightgreen", "salmon"])
                ax3.set_title("Macro Average Metrics", fontweight="bold")
                ax3.set_ylabel("Score")
                ax3.set_ylim(0, 1)

                for bar, value in zip(bars, values):
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
        else:
            ax3.text(
                0.5,
                0.5,
                "Evaluation metrics\nnot available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
            )
            ax3.set_title("Evaluation Metrics", fontweight="bold")

        # 4. Data statistics
        total_samples = len(df)
        num_features = len([col for col in df.columns if col not in [
                           "Label", "Label_Original"]])
        num_classes = df[label_col].nunique()

        stats_text = f"""Dataset Statistics:
        
• Total Samples: {total_samples:,}
• Number of Features: {num_features}
• Number of Classes: {num_classes}
• Most Common Class: {label_counts.index[0]}
• Class Imbalance Ratio: {label_counts.iloc[0] / label_counts.iloc[-1]:.1f}:1"""

        ax4.text(
            0.1,
            0.5,
            stats_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        ax4.set_title("Dataset Overview", fontweight="bold")
        ax4.axis("off")

        plt.suptitle("RL-IDS Summary Dashboard",
                     fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.success(f"✅ Saved summary dashboard to {save_path}")

    except Exception as e:
        logger.error(f"Error creating summary dashboard: {e}")


@app.command()
def main(
    input_path: Path = typer.Option(
        BALANCED_DATA_FILE, help="Path to dataset CSV"),
    output_path: Path = typer.Option(
        FIGURES_DIR / "class_distribution.png", help="Output plot path"
    ),
):
    """Generate class distribution plot (default behavior for backward compatibility)."""
    plot_label_distribution(input_path, output_path)


if __name__ == "__main__":
    app()

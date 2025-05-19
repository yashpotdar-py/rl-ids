"""
Evaluation script for reinforcement learning-based intrusion detection models.

This module provides comprehensive evaluation procedures for trained DQN and 
Policy Gradient agents on the network intrusion detection environment with 
detailed metrics reporting and MLflow integration.
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
import mlflow
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from rl_ids.environment import IntrusionEnv
from rl_ids.modeling.dqn_agent import DQNAgent
from rl_ids.modeling.pg_agent import PGAgent
from rl_ids.config import PROCESSED_DATA_DIR, MODELS_DIR


def setup_logger(log_file: Optional[str] = None):
    """Configure logger with appropriate settings."""
    logger.remove()  # Remove default handler

    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, flush=True),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>: <level>{message}</level>"
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            sink=log_file,
            level="DEBUG",
            rotation="10 MB",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}: {message}",
            backtrace=True,
            diagnose=True
        )


def evaluate_agent(
    env: IntrusionEnv,
    agent: Union[DQNAgent, PGAgent],
    num_episodes: int = 5,
    render: bool = False,
    detailed: bool = False,
    progress_bar: bool = True,
    aggregate_results: bool = False
) -> Dict[str, Any]:
    """
    Evaluate agent performance on the environment.

    Args:
        env: Environment to evaluate on
        agent: Agent to evaluate (DQN or PG)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        detailed: Whether to return detailed metrics
        progress_bar: Whether to display a progress bar
        aggregate_results: Whether to aggregate results across episodes

    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    metrics_list = []
    y_true_all = []
    y_pred_all = []
    confidence_all = []

    episode_range = tqdm(range(num_episodes)) if progress_bar else range(num_episodes)

    for i in episode_range:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        y_true = []
        y_pred = []
        confidence = []

        while not done:
            # Get action based on agent type
            if isinstance(agent, DQNAgent):
                # For DQN, get action with epsilon=0 (no exploration)
                action = agent.select_action(state, epsilon=0.0)

                # If we need confidence scores, get Q-values
                if detailed:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(np.array([state])).to(agent.device)
                        q_values = agent.q_net(state_tensor).cpu().numpy()[0]
                        confidence.append(q_values)
            else:
                # For PG, use deterministic action selection
                action = agent.select_action(state, deterministic=True)

            # Execute action
            next_state, reward, done, _, info = env.step(action)

            if render:
                env.render()

            # Track true label (from environment)
            y_true.append(info.get("label", None))
            y_pred.append(action)

            episode_reward += reward
            state = next_state

        # Collect metrics
        rewards.append(episode_reward)
        metrics_list.append(env.get_episode_metrics())

        # Collect predictions for aggregated analysis
        if aggregate_results:
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            if detailed and isinstance(agent, DQNAgent):
                confidence_all.extend(confidence)

    # Calculate average metrics across episodes
    avg_metrics = {
        "reward": np.mean(rewards),
        "accuracy": np.mean([m.accuracy for m in metrics_list]),
        "precision": np.mean([m.precision for m in metrics_list]),
        "recall": np.mean([m.recall for m in metrics_list]),
        "f1_score": np.mean([m.f1_score for m in metrics_list]),
        "true_positives": sum(m.true_positives for m in metrics_list),
        "false_positives": sum(m.false_positives for m in metrics_list),
        "true_negatives": sum(m.true_negatives for m in metrics_list),
        "false_negatives": sum(m.false_negatives for m in metrics_list),
    }

    # Add detailed metrics if requested
    if detailed and aggregate_results and y_true_all:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_all, y_pred_all)
        avg_metrics["confusion_matrix"] = cm.tolist()

        # Calculate classification report
        class_names = ["Benign", "Attack"]
        avg_metrics["classification_report"] = classification_report(
            y_true_all, y_pred_all, target_names=class_names, output_dict=True
        )

        # For DQN, we can calculate ROC and PR curves from confidences
        if isinstance(agent, DQNAgent) and confidence_all:
            # Convert confidences to attack probability (softmax of q-values)
            attack_probs = np.array(confidence_all)[:, 1]

            # Calculate ROC curve and AUC
            if len(np.unique(y_true_all)) > 1:  # Only if we have both classes
                fpr, tpr, _ = roc_curve(y_true_all, attack_probs)
                roc_auc = auc(fpr, tpr)
                avg_metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                avg_metrics["roc_auc"] = roc_auc

                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_true_all, attack_probs)
                pr_auc = auc(recall, precision)
                avg_metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
                avg_metrics["pr_auc"] = pr_auc

    return avg_metrics


def load_agent(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    agent_type: str = "dqn",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    trust_checkpoint: bool = False
) -> Union[DQNAgent, PGAgent]:
    """
    Load a trained agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        agent_type: Type of agent ('dqn' or 'pg')
        device: Device to load model on
        trust_checkpoint: Whether to trust the checkpoint source

    Returns:
        Loaded agent
    """
    logger.info(f"Loading {agent_type.upper()} agent from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if agent_type.lower() == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],  # Default architecture
            device=device
        )

        # Direct loading approach if we trust the checkpoint source
        if trust_checkpoint:
            logger.info("Loading checkpoint with weights_only=False (trusted source)")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                # Examine checkpoint structure first
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Successfully loaded model_state_dict")
                elif 'state_dict' in checkpoint:
                    agent.q_net.load_state_dict(checkpoint['state_dict'])
                    logger.info("Successfully loaded state_dict")
                elif 'q_net_state_dict' in checkpoint:
                    agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
                    logger.info("Successfully loaded q_net_state_dict")
                # Direct state dict (no wrapper dictionary)
                elif all(k.endswith(('.weight', '.bias')) for k in list(checkpoint.keys())[:2]):
                    agent.q_net.load_state_dict(checkpoint)
                    logger.info("Successfully loaded direct state dict")
                else:
                    logger.warning(f"Unknown checkpoint format with keys: {list(checkpoint.keys())}")
                    raise ValueError("Checkpoint format not recognized")

                return agent
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=False: {str(e)}")
                # Continue to try other methods

        # Simplified final approach for model loading
        try:
            logger.info("Attempting simplified model loading")

            # Manual approach to extract state dict only
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp_path = tmp.name

            # Copy the model weights to a simpler file structure
            if trust_checkpoint:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                # Try to find the state dict in various formats
                state_dict = None
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'q_net_state_dict' in checkpoint:
                    state_dict = checkpoint['q_net_state_dict']

                if state_dict:
                    # Save only the state_dict to avoid serialization issues
                    torch.save(state_dict, tmp_path)

                    # Load the clean state dict
                    agent.q_net.load_state_dict(torch.load(tmp_path, map_location=device))
                    logger.info("Successfully loaded model via temporary file")

                    # Clean up
                    os.unlink(tmp_path)
                    return agent
        except Exception as e:
            logger.warning(f"Failed simplified loading approach: {str(e)}")

        # If we reached here, we couldn't load the model
        raise RuntimeError(
            "Failed to load model checkpoint. Try running with --trust_checkpoint flag "
            "if you trust the source of this checkpoint."
        )

    elif agent_type.lower() == "pg":
        agent = PGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],  # Default architecture
            device=device
        )

        # Direct loading approach if we trust the checkpoint source
        if trust_checkpoint:
            logger.info("Loading checkpoint with weights_only=False (trusted source)")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                # Examine checkpoint structure first
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Successfully loaded model_state_dict for policy network")
                elif 'state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['state_dict'])
                    logger.info("Successfully loaded state_dict for policy network")
                elif 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    logger.info("Successfully loaded policy_net_state_dict")
                # Direct state dict (no wrapper dictionary)
                elif all(k.endswith(('.weight', '.bias')) for k in list(checkpoint.keys())[:2]):
                    agent.policy_net.load_state_dict(checkpoint)
                    logger.info("Successfully loaded direct state dict for policy network")
                else:
                    logger.warning(f"Unknown checkpoint format with keys: {list(checkpoint.keys())}")
                    raise ValueError("Checkpoint format not recognized")

                return agent
            except Exception as e:
                logger.warning(f"Failed to load PG agent with weights_only=False: {str(e)}")

        # If we reached here, we couldn't load the model
        raise RuntimeError(
            "Failed to load policy model checkpoint. Try running with --trust_checkpoint flag "
            "if you trust the source of this checkpoint."
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def generate_plots(metrics: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Generate evaluation plots and save them to disk.

    Args:
        metrics: Evaluation metrics including data for plots
        output_dir: Directory to save plots to

    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # Confusion matrix plot
    if "confusion_matrix" in metrics:
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        plot_paths["confusion_matrix"] = cm_path

    # ROC curve plot
    if "roc_curve" in metrics:
        plt.figure(figsize=(8, 6))
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        roc_auc = metrics.get("roc_auc", 0)

        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (area = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight", dpi=300)
        plt.close()
        plot_paths["roc_curve"] = roc_path

    # Precision-Recall curve plot
    if "pr_curve" in metrics:
        plt.figure(figsize=(8, 6))
        precision = metrics["pr_curve"]["precision"]
        recall = metrics["pr_curve"]["recall"]
        pr_auc = metrics.get("pr_auc", 0)

        plt.plot(recall, precision, color="blue", lw=2,
                 label=f"PR curve (area = {pr_auc:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        pr_path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.savefig(pr_path, bbox_inches="tight", dpi=300)
        plt.close()
        plot_paths["pr_curve"] = pr_path

    return plot_paths


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RL agents for intrusion detection")

    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to agent checkpoint file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned.parquet"),
        help="Path to processed data file for evaluation"
    )

    # Agent parameters
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["dqn", "pg"],
        default="dqn",
        help="Type of agent to evaluate"
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of samples (for faster evaluation)"
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        help="Randomly sample from dataset"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results and plots"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="evaluation",
        help="Experiment name for MLflow"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path"
    )
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Track evaluation with MLflow"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed metrics and plots"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help="Trust checkpoint source and load with weights_only=False"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(args.log_file)

    # Determine output directory
    if args.output_dir is None:
        base_dir = os.path.join(MODELS_DIR, f"evaluation_{args.agent_type}")

        # Check if directory exists and create a numbered version if needed
        if os.path.exists(base_dir):
            counter = 1
            while os.path.exists(f"{base_dir}_{counter}"):
                counter += 1
            args.output_dir = f"{base_dir}_{counter}"
        else:
            args.output_dir = base_dir

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Log arguments
        logger.info(f"Starting evaluation: {args.experiment_name}")
        logger.info(f"Arguments: {args}")

        # Load data to get dimensions
        logger.info(f"Loading data from {args.data_path}")
        try:
            df = pd.read_parquet(args.data_path)
            if "Label" not in df.columns:
                raise ValueError("Data must contain a 'Label' column")

            state_dim = df.shape[1] - 1  # Exclude label column
            action_dim = 2  # Binary classification (allow/block)

            logger.info(f"Data loaded: {df.shape[0]} samples, {state_dim} features")
            logger.info(f"Class distribution: {df['Label'].value_counts().to_dict()}")

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

        # Initialize environment
        logger.info("Initializing environment")
        env = IntrusionEnv(
            data_path=args.data_path,
            sample_limit=args.sample_limit,
            random_sampling=args.random_sampling
        )

        # Load agent
        agent = load_agent(
            checkpoint_path=args.checkpoint_path,
            state_dim=state_dim,
            action_dim=action_dim,
            agent_type=args.agent_type,
            device=args.device,
            trust_checkpoint=args.trust_checkpoint
        )

        # Evaluate agent
        logger.info(f"Evaluating agent on {args.episodes} episodes")
        start_time = time.time()

        metrics = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            render=args.render,
            detailed=args.detailed,
            progress_bar=True,
            aggregate_results=True
        )

        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")

        # Process and save results
        metrics["evaluation_time"] = evaluation_time

        # Generate and save plots if detailed metrics are requested
        plot_paths = {}
        if args.detailed:
            logger.info("Generating evaluation plots")
            plot_paths = generate_plots(metrics, args.output_dir)

            # Update metrics with plot paths
            metrics["plot_paths"] = plot_paths

        # Save metrics to JSON
        metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")

        # Remove non-serializable items before saving
        serializable_metrics = {k: v for k, v in metrics.items()
                                if k not in ["confusion_matrix", "classification_report",
                                             "roc_curve", "pr_curve", "plot_paths"]}

        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

        # Print key metrics
        logger.info(f"Key metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  TP/FP/TN/FN: {metrics['true_positives']}/{metrics['false_positives']}/"
                    f"{metrics['true_negatives']}/{metrics['false_negatives']}")

        # Log to MLflow if requested
        if args.use_mlflow:
            logger.info("Logging results to MLflow")

            with mlflow.start_run(run_name=args.experiment_name):
                # Log parameters
                mlflow.log_params({
                    "agent_type": args.agent_type,
                    "checkpoint_path": os.path.basename(args.checkpoint_path),
                    "data_path": os.path.basename(args.data_path),
                    "episodes": args.episodes,
                    "sample_limit": args.sample_limit,
                    "random_sampling": args.random_sampling,
                })

                # Log metrics
                for k, v in serializable_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                # Log plots
                for plot_name, plot_path in plot_paths.items():
                    mlflow.log_artifact(plot_path)

                # Log model
                if isinstance(agent, DQNAgent):
                    mlflow.pytorch.log_model(agent.q_net, "model")
                elif isinstance(agent, PGAgent):
                    mlflow.pytorch.log_model(agent.policy_net, "model")

                logger.info(f"MLflow tracking completed - run_id: {mlflow.active_run().info.run_id}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.exception("Exception details:")
        raise

    logger.info(f"Evaluation {args.experiment_name} completed successfully")


if __name__ == "__main__":
    main()

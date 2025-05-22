"""
Training script for reinforcement learning-based intrusion detection models.

This module provides training procedures for DQN and Policy Gradient agents
on the network intrusion detection environment with Prometheus monitoring.
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from loguru import logger
import numpy as np
from prometheus_client import Counter, Gauge, Summary, start_http_server
import torch
from tqdm import trange

from rl_ids.config import MODELS_DIR, PROCESSED_DATA_DIR
from rl_ids.environment import IntrusionEnv
from rl_ids.modeling.dqn_agent import DQNAgent
from rl_ids.modeling.pg_agent import PGAgent

# Prometheus metrics
EPISODE_REWARD = Summary('episode_reward', 'Total reward per episode')
EPISODE_DURATION = Summary('episode_duration', 'Duration of episode in seconds')
TRAINING_LOSS = Summary('training_loss', 'Training loss per optimization step')
DETECTIONS = Counter('detections_total', 'Number of correct detections')
FALSE_ALARMS = Counter('false_alarms_total', 'Number of false alarms')
EPSILON = Gauge('exploration_epsilon', 'Current epsilon value for exploration')
MEMORY_USAGE = Gauge('memory_usage_mb', 'Current GPU memory usage in MB')
LEARNING_RATE = Gauge('learning_rate', 'Current learning rate')


def setup_logger(log_file: Optional[str] = None):
    """
    Configure the logger for the training process.

    Args:
        log_file: Path to log file (if None, logs to stdout only)
    """
    logger.remove()  # Remove default handler

    # Add stdout handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Add file handler if specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG"
        )


def update_memory_metrics():
    """Update GPU memory usage metrics if CUDA is available."""
    if torch.cuda.is_available():
        # Get current GPU memory usage in MB
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        MEMORY_USAGE.set(memory_allocated)

        if memory_reserved > memory_allocated * 2:
            # Memory fragmentation might be occurring
            logger.warning(
                f"High memory reservation detected: {memory_reserved:.1f}MB allocated, "
                f"{memory_reserved:.1f}MB reserved"
            )
            # Force garbage collection and empty cache
            torch.cuda.empty_cache()


def evaluate_agent(
    env: IntrusionEnv,
    agent: DQNAgent,
    num_episodes: int = 5,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate agent performance on the environment.

    Args:
        env: Environment to evaluate on
        agent: Agent to evaluate
        num_episodes: Number of episodes to run
        render: Whether to render the environment

    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    metrics = []

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # No exploration during evaluation
            next_state, reward, done, _, _ = env.step(action)
            if render:
                env.render()

            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        metrics.append(env.get_episode_metrics())

    # Calculate average metrics
    avg_metrics = {
        "reward": np.mean(rewards),
        "accuracy": np.mean([m.accuracy for m in metrics]),
        "precision": np.mean([m.precision for m in metrics]),
        "recall": np.mean([m.recall for m in metrics]),
        "f1_score": np.mean([m.f1_score for m in metrics]),
    }

    return avg_metrics


def train_dqn(
    env: IntrusionEnv,
    agent: DQNAgent,
    episodes: int,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    eval_interval: int = 10,
    checkpoint_interval: int = 50,
    checkpoint_dir: Optional[str] = None,
    experiment_name: str = "dqn_experiment",
    max_steps: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """
    Train DQN agent on the environment with Prometheus monitoring.

    Args:
        env: Environment to train on
        agent: DQN agent to train
        episodes: Number of episodes to train for
        eps_start: Starting epsilon value for exploration
        eps_end: Minimum epsilon value
        eps_decay: Epsilon decay factor
        eval_interval: Episodes between evaluations
        checkpoint_interval: Episodes between checkpoints
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of this experiment
        max_steps: Maximum steps per episode

    Returns:
        Dictionary containing training history
    """
    logger.info(f"Starting DQN training for {episodes} episodes")
    start_time = time.time()

    # History tracking
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses": [],
        "eval_metrics": [],
        "epsilon": []
    }

    # Set up checkpoint directory
    if checkpoint_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    eps = eps_start
    episode_rewards = []

    # Update epsilon gauge
    EPSILON.set(eps)

    progress_bar = trange(episodes, desc="Training DQN")
    for episode in progress_bar:
        episode_start_time = time.time()

        # Reset environment with max_steps option
        state, _ = env.reset(options={"max_steps": max_steps})
        done = False
        episode_reward = 0
        episode_length = 0
        episode_loss = []

        # Episode loop
        while not done:
            # Select and execute action
            action = agent.select_action(state, eps)
            next_state, reward, done, _, _ = env.step(action)

            # Update Prometheus metrics for detections
            if reward > 0:
                DETECTIONS.inc()
            else:
                FALSE_ALARMS.inc()

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Optimize agent
            loss = agent.optimize()
            if loss is not None:
                episode_loss.append(loss)
                TRAINING_LOSS.observe(loss)

            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Clear unnecessary tensors from GPU memory
            if torch.cuda.is_available() and episode_length % 100 == 0:
                update_memory_metrics()

        # Decay epsilon
        eps = max(eps_end, eps * eps_decay)
        EPSILON.set(eps)

        # Update history
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(episode_length)
        history["epsilon"].append(eps)
        if episode_loss:
            history["losses"].append(np.mean(episode_loss))

        # Update Prometheus episode metrics
        EPISODE_REWARD.observe(episode_reward)
        EPISODE_DURATION.observe(time.time() - episode_start_time)

        # Update progress bar
        progress_bar.set_postfix(reward=f"{episode_reward:.1f}", eps=f"{eps:.2f}")

        # Track rewards for recent episodes
        episode_rewards.append(episode_reward)
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_metrics = evaluate_agent(env, agent)
            history["eval_metrics"].append({
                "episode": episode + 1,
                **eval_metrics
            })

            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                f"Eval: Acc={eval_metrics['accuracy']:.4f}, "
                f"F1={eval_metrics['f1_score']:.4f}, "
                f"P={eval_metrics['precision']:.4f}, "
                f"R={eval_metrics['recall']:.4f}"
            )

        # Save checkpoint periodically
        if checkpoint_dir and (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode + 1}.tar")
            agent.save_checkpoint(
                episode=episode + 1,
                epsilon=eps,
                rewards=episode_rewards,
                path=checkpoint_path
            )

            # Save training history
            with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "final_model.tar")
        agent.save_checkpoint(
            episode=episodes,
            epsilon=eps,
            rewards=episode_rewards,
            path=final_path
        )

    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    return history


def train_pg(
    env: IntrusionEnv,
    agent: PGAgent,
    episodes: int,
    eval_interval: int = 10,
    checkpoint_interval: int = 50,
    checkpoint_dir: Optional[str] = None,
    experiment_name: str = "pg_experiment",
) -> Dict[str, List[Any]]:
    """
    Train Policy Gradient agent on the environment with Prometheus monitoring.

    Args:
        env: Environment to train on
        agent: PG agent to train
        episodes: Number of episodes to train for
        eval_interval: Episodes between evaluations
        checkpoint_interval: Episodes between checkpoints
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of this experiment

    Returns:
        Dictionary containing training history
    """
    logger.info(f"Starting Policy Gradient training for {episodes} episodes")
    start_time = time.time()

    # History tracking
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "eval_metrics": []
    }

    # Set up checkpoint directory
    if checkpoint_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    episode_rewards = []

    progress_bar = trange(episodes, desc="Training PG")
    for episode in progress_bar:
        episode_start_time = time.time()

        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # Episode loop - collect experience
        while not done:
            # Select and execute action
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # Update Prometheus metrics for detections
            if reward > 0:
                DETECTIONS.inc()
            else:
                FALSE_ALARMS.inc()

            # Store transition
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Clear unnecessary tensors from GPU memory
            if torch.cuda.is_available() and episode_length % 100 == 0:
                update_memory_metrics()

        # Perform update at end of episode
        update_stats = agent.update()

        # Update Prometheus metrics
        EPISODE_REWARD.observe(episode_reward)
        EPISODE_DURATION.observe(time.time() - episode_start_time)
        TRAINING_LOSS.observe(update_stats["total_loss"])

        # Update history
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(episode_length)
        history["policy_losses"].append(update_stats["policy_loss"])
        history["value_losses"].append(update_stats["value_loss"])
        history["entropies"].append(update_stats["entropy"])

        # Update progress bar
        progress_bar.set_postfix(reward=f"{episode_reward:.1f}", loss=f"{update_stats['total_loss']:.3f}")

        # Track rewards for recent episodes
        episode_rewards.append(episode_reward)
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_metrics = evaluate_agent(env, agent, num_episodes=5)
            history["eval_metrics"].append({
                "episode": episode + 1,
                **eval_metrics
            })

            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                f"Eval: Acc={eval_metrics['accuracy']:.4f}, "
                f"F1={eval_metrics['f1_score']:.4f}, "
                f"P={eval_metrics['precision']:.4f}, "
                f"R={eval_metrics['recall']:.4f}"
            )

        # Save checkpoint periodically
        if checkpoint_dir and (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode + 1}.tar")
            agent.save_checkpoint(
                episode=episode + 1,
                rewards=episode_rewards,
                path=checkpoint_path
            )

            # Save training history
            with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "final_model.tar")
        agent.save_checkpoint(
            episode=episodes,
            rewards=episode_rewards,
            path=final_path
        )

    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    return history


def parse_arguments():
    """Parse command line arguments for training script."""
    parser = argparse.ArgumentParser(description="Train RL agents for intrusion detection")

    # Environment parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned.parquet"),
        help="Path to processed data file"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of samples (for faster testing)"
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        help="Randomly sample from dataset"
    )

    # Training parameters
    parser.add_argument(
        "--method",
        type=str,
        choices=["dqn", "pg"],
        default="dqn",
        help="RL method to use (dqn or pg)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    # DQN specific parameters
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100000,
        help="Replay buffer size"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--eps_start",
        type=float,
        default=1.0,
        help="Starting epsilon for exploration"
    )
    parser.add_argument(
        "--eps_end",
        type=float,
        default=0.01,
        help="Ending epsilon for exploration"
    )
    parser.add_argument(
        "--eps_decay",
        type=float,
        default=0.995,
        help="Epsilon decay rate"
    )
    parser.add_argument(
        "--double_dqn",
        action="store_true",
        help="Use double DQN algorithm"
    )

    # Experiment parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment (default: auto-generated)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Episode interval between evaluations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--prometheus_port",
        type=int,
        default=8000,
        help="Port for Prometheus metrics server"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create experiment name if not provided
    if not args.experiment_name:
        # Use method name without timestamp
        args.experiment_name = args.method

        # Check if experiment directory already exists and add suffix if needed
        base_name = args.experiment_name
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        counter = 1

        while os.path.exists(checkpoint_dir):
            args.experiment_name = f"{base_name}_{counter}"
            checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
            counter += 1

    # Setup logging
    log_file = os.path.join(args.log_dir, f"{args.experiment_name}.log")
    setup_logger(log_file)

    # Start Prometheus metrics server
    try:
        start_http_server(args.prometheus_port)
        logger.info(f"Prometheus metrics exposed on port {args.prometheus_port}")
    except Exception as e:
        logger.warning(f"Failed to start Prometheus metrics server: {e}")

    # Log arguments
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Arguments: {args}")

    # Initialize environment
    logger.info(f"Initializing environment with data from {args.data_path}")
    env = IntrusionEnv(
        data_path=args.data_path,
        sample_limit=args.sample_limit,
        random_sampling=args.random_sampling
    )

    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)

    if args.method == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            lr=args.lr,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            double_dqn=args.double_dqn,
            device=args.device,
            checkpoint_dir=checkpoint_dir
        )

        # Set learning rate gauge
        current_lr = agent.scheduler.get_last_lr()[0]
        LEARNING_RATE.set(current_lr)

        # Train agent
        history = train_dqn(
            env=env,
            agent=agent,
            episodes=args.episodes,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            eval_interval=args.eval_interval,
            checkpoint_interval=20,
            checkpoint_dir=checkpoint_dir,
            experiment_name=args.experiment_name,
            max_steps=args.max_steps  # Pass this parameter
        )

    elif args.method == "pg":
        agent = PGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            lr_policy=args.lr,
            lr_value=args.lr * 3,  # Value function often benefits from higher learning rate
            gamma=args.gamma,
            device=args.device,
            checkpoint_dir=checkpoint_dir
        )

        # Train agent
        history = train_pg(
            env=env,
            agent=agent,
            episodes=args.episodes,
            eval_interval=args.eval_interval,
            checkpoint_interval=20,
            checkpoint_dir=checkpoint_dir,
            experiment_name=args.experiment_name
        )

        print(history)

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    # Final evaluation
    logger.info("Performing final evaluation")
    final_metrics = evaluate_agent(env, agent, num_episodes=10, render=False)
    logger.info(f"Final metrics: {final_metrics}")

    # Save final metrics
    metrics_file = os.path.join(checkpoint_dir, "final_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Clean up any remaining resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Experiment {args.experiment_name} completed successfully")


if __name__ == "__main__":
    main()

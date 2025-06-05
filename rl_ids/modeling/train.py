from rl_ids.agents.dqn_agent import DQNConfig, DQNAgent
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.config import NORMALISED_DATA_FILE, MODELS_DIR, REPORTS_DIR, EPISODES_DIR
import os
import pandas as pd
import numpy as np
import typer
from tqdm import tqdm
from pathlib import Path
import sys

from loguru import logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Only show INFO and above


app = typer.Typer()


@app.command()
def main(
    data_path: Path = NORMALISED_DATA_FILE,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    episodes_dir: Path = EPISODES_DIR,
    num_episodes: int = typer.Option(200, help="Number of training episodes"),
    target_update_interval: int = typer.Option(
        5, help="Target network update interval"),
    lr: float = typer.Option(5e-5, help="Learning Rate"),
    gamma: float = typer.Option(0.99, help="Discount Factor"),
    epsilon: float = typer.Option(1.0, help="Initial Exploration Rate"),
    eps_decay: float = typer.Option(0.998, help="Epsilon Decay Rate"),
    eps_min: float = typer.Option(0.05, help="Minimum Epsilon Value"),
    memory_size: int = typer.Option(50000, help="Replay Buffer Size"),
    batch_size: int = typer.Option(128, help="Training Batch Size"),
    save_interval: int = typer.Option(20, help="Model Save Interval"),
    max_steps_per_episode: int = typer.Option(
        10000, help="Max steps per episode")
):
    """Train DQN Agent on IDS Detection Task"""
    logger.info("Starting DQN training for IDS Detection")

    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Get feature columns (exclude label columns)
    feature_cols = [col for col in df.columns if col not in [
        'Label', 'Label_Original']]

    # Get dimensions
    input_dim = len(feature_cols)
    n_classes = len(np.unique(df['Label'].values))

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Input Dimension: {input_dim}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class distribution: \n{df['Label'].value_counts()}")

    # Add timeout or sample limiting for episodes
    max_steps_per_episode = max_steps_per_episode  # Limit steps per episode # DEBUG
    logger.info(f"Max steps per episode: {max_steps_per_episode}")
    logger.info(f"Starting training for {num_episodes} episodes...")

    # Initialize environment
    logger.info("Initializing Environment...")
    env = IDSDetectionEnv(data_path=data_path,
                          feature_cols=feature_cols, label_col="Label")

    # Initialize agent with configuration
    logger.info("Initializing DQN Agent...")
    config = DQNConfig(
        state_dim=input_dim,
        action_dim=n_classes,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        eps_decay=eps_decay,
        eps_min=eps_min,
        memory_size=memory_size,
        batch_size=batch_size,
        hidden_dims=[256, 128, 64]  # Adjusted for IDS tasks
    )
    agent = DQNAgent(config=config)

    # Create Directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)

    # Training metrics
    rewards_per_episodes = []
    losses_per_episodes = []
    accuracies_per_episodes = []

    logger.info(f"Starting training for {num_episodes} episodes...")

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state, info = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        correct_predictions = 0
        total_predictions = 0
        # DEBUG
        step_count = 0  # DEBUG
        while not done and step_count < max_steps_per_episode:  # DEBUG
            # while not done:
            step_count += 1  # DEBUG
            # Agent selects action
            action = agent.act(state=state, training=True)

            # Environment step
            next_state, reward, done, truncated, info = env.step(action=action)

            # Store experience in replay buffer
            agent.remember(state, action, reward,
                           next_state, done or truncated)

            # Train agent (if enough experiences in buffer)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)

            # Track Accuracy
            actual_label = info.get("actual_label", -1)
            if actual_label != -1:
                total_predictions += 1
                if action == actual_label:
                    correct_predictions += 1

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        # Update target network periodically
        if episode % target_update_interval == 0:
            agent.update_target()
            logger.info(f"Updated target network at episode {episode}")

        # Calculate episode metrics
        episode_accuracy = correct_predictions / max(total_predictions, 1)
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0

        # Store metrics
        rewards_per_episodes.append(total_reward)
        losses_per_episodes.append(avg_episode_loss)
        accuracies_per_episodes.append(episode_accuracy)

        # Log progress
        if episode % 10 == 0 or episode == num_episodes - 1:
            logger.info(
                f"Episode {episode+1}/{num_episodes} - "
                f"Reward: {total_reward:.2f} - "
                f"Accuracy: {episode_accuracy:.4f} - "
                f"Loss: {avg_episode_loss:.4f} - "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        # Save the model periodically
        if episode % save_interval == 0 and episode > 0:
            model_path = episodes_dir / f"dqn_model_episode_{episode}.pt"
            agent.save_model(model_path)
            logger.info(f"Model saved at episode {episode}")

    # Save final model
    final_model_path = models_dir / "dqn_model_final.pt"
    agent.save_model(final_model_path)
    logger.success(f"✅ Final model saved at: {final_model_path}")

    # Save training metrics
    metrics_df = pd.DataFrame({
        "Episode": range(1, num_episodes + 1),
        "Reward": rewards_per_episodes,
        "Loss": losses_per_episodes,
        "Accuracy": accuracies_per_episodes
    })

    metrics_path = reports_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.success(f"📈 Training metrics saved to: {metrics_path}")

    # Print final statistics
    final_avg_reward = np.mean(rewards_per_episodes[-10:])  # Last 10 episodes
    final_avg_accuracy = np.mean(accuracies_per_episodes[-10:])
    max_accuracy = np.max(accuracies_per_episodes)

    logger.info("Training completed!")
    logger.info(
        f"Final average reward (last 10 episodes): {final_avg_reward:.2f}")
    logger.info(
        f"Final average accuracy (last 10 episodes): {final_avg_accuracy:.4f}")
    logger.info(f"Maximum accuracy achieved: {max_accuracy:.4f}")

    return {
        "final_avg_reward": final_avg_reward,
        "final_avg_accuracy": final_avg_accuracy,
        "max_accuracy": max_accuracy,
        "model_path": str(final_model_path),
        "metrics_path": str(metrics_path)
    }


if __name__ == "__main__":
    app()

import os
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer
import torch

from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.config import EPISODES_DIR, MODELS_DIR, REPORTS_DIR, TRAIN_DATA_FILE, VAL_DATA_FILE
from rl_ids.environments.ids_env import IDSDetectionEnv


app = typer.Typer()


@app.command()
def main(
    train_data_path: Path = typer.Option(
        TRAIN_DATA_FILE, help="Path to training data"
    ),
    val_data_path: Path = typer.Option(
        VAL_DATA_FILE, help="Path to validation data"
    ),
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    episodes_dir: Path = EPISODES_DIR,
    num_episodes: int = typer.Option(500, help="Number of training episodes"),
    target_update_interval: int = typer.Option(
        10, help="Target network update interval"
    ),
    lr: float = typer.Option(1e-4, help="Learning Rate"),
    lr_scheduler: str = typer.Option(
        "cosine", help="Learning rate scheduler: cosine, step, or none"),
    gamma: float = typer.Option(0.995, help="Discount Factor"),
    epsilon: float = typer.Option(1.0, help="Initial Exploration Rate"),
    eps_decay: float = typer.Option(0.9995, help="Epsilon Decay Rate"),
    eps_min: float = typer.Option(0.01, help="Minimum Epsilon Value"),
    memory_size: int = typer.Option(100000, help="Replay Buffer Size"),
    batch_size: int = typer.Option(256, help="Training Batch Size"),
    warmup_steps: int = typer.Option(
        1000, help="Warmup steps before training"),
    hidden_dims: str = typer.Option(
        "1024,512,256,128", help="Hidden layer dimensions (comma-separated)"),
    dropout_rate: float = typer.Option(
        0.2, help="Dropout rate for regularization"),
    use_layer_norm: bool = typer.Option(True, help="Use layer normalization"),
    save_interval: int = typer.Option(50, help="Model Save Interval"),
    max_steps_per_episode: int = typer.Option(
        15000, help="Max steps per episode"
    ),
    validation_interval: int = typer.Option(
        5, help="Validation evaluation interval"
    ),
    early_stopping_patience: int = typer.Option(
        30, help="Early stopping patience (episodes)"
    ),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping value"),
    weight_decay: float = typer.Option(
        1e-5, help="Weight decay for regularization"),
    prioritized_replay: bool = typer.Option(
        False, help="Use prioritized experience replay"),
    double_dqn: bool = typer.Option(True, help="Use Double DQN"),
    dueling: bool = typer.Option(True, help="Use Dueling DQN architecture"),
    curriculum_learning: bool = typer.Option(
        True, help="Use curriculum learning"),
    curriculum_stages: int = typer.Option(
        3, help="Number of curriculum stages"),
):
    """Train Enhanced DQN Agent on IDS Detection Task with advanced optimizations"""
    start_time = time.time()
    logger.info("🚀 Starting Enhanced DQN training for IDS Detection")
    logger.info("=" * 70)

    # Check if data files exist
    if not train_data_path.exists():
        logger.error(f"❌ Training data not found: {train_data_path}")
        logger.info(
            "💡 Please run 'python -m rl_ids.make_dataset' first to generate processed data")
        raise typer.Exit(1)

    use_validation = val_data_path.exists()
    if not use_validation:
        logger.warning(f"⚠️  Validation data not found: {val_data_path}")
        logger.info("🔄 Training without validation monitoring")

    # Load and prepare training data
    logger.info(f"📂 Loading training data from {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    # Load validation data if available
    val_df = None
    if use_validation:
        logger.info(f"📂 Loading validation data from {val_data_path}")
        val_df = pd.read_csv(val_data_path)

    # Get feature columns (exclude label columns)
    feature_cols = [col for col in train_df.columns if col not in [
        "Label", "Label_Original"]]

    # Get dimensions
    input_dim = len(feature_cols)
    n_classes = len(np.unique(train_df["Label"].values))

    logger.info(f"📊 Training dataset shape: {train_df.shape}")
    if use_validation:
        logger.info(f"📊 Validation dataset shape: {val_df.shape}")
    logger.info(f"🔢 Input dimension: {input_dim}")
    logger.info(f"🏷️  Number of classes: {n_classes}")

    # Log class distributions
    logger.info("📈 Training class distribution:")
    train_dist = train_df['Label'].value_counts().sort_index()
    for label, count in train_dist.items():
        percentage = count / len(train_df) * 100
        logger.info(
            f"   Class {label}: {count:8,} samples ({percentage:5.1f}%)")

    if use_validation:
        logger.info("📈 Validation class distribution:")
        val_dist = val_df['Label'].value_counts().sort_index()
        for label, count in val_dist.items():
            percentage = count / len(val_df) * 100
            logger.info(
                f"   Class {label}: {count:8,} samples ({percentage:5.1f}%)")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    if device.type == "cuda":
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

    # Parse hidden dimensions
    hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(",")]
    logger.info(
        f"🏗️  Network architecture: {input_dim} → {' → '.join(map(str, hidden_dims_list))} → {n_classes}")

    # Initialize environments
    logger.info("🌍 Initializing training environment...")
    train_env = IDSDetectionEnv(
        data_path=train_data_path,
        feature_cols=feature_cols,
        label_col="Label"
    )

    val_env = None
    if use_validation:
        logger.info("🌍 Initializing validation environment...")
        val_env = IDSDetectionEnv(
            data_path=val_data_path,
            feature_cols=feature_cols,
            label_col="Label"
        )

    # Initialize enhanced agent with configuration
    logger.info("🤖 Initializing Enhanced DQN Agent...")
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
        hidden_dims=hidden_dims_list,
        device=device.type,
        # Enhanced parameters
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
        weight_decay=weight_decay,
        double_dqn=double_dqn,
        dueling=dueling,
        prioritized_replay=prioritized_replay,
    )
    agent = DQNAgent(config=config)

    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler != "none":
        if lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                agent.optimizer, T_max=num_episodes, eta_min=lr * 0.01
            )
            logger.info(f"📉 Using Cosine Annealing LR scheduler")
        elif lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                agent.optimizer, step_size=num_episodes // 4, gamma=0.5
            )
            logger.info(f"📉 Using Step LR scheduler")

    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)

    # Training metrics
    train_rewards = []
    train_losses = []
    train_accuracies = []
    val_rewards = []
    val_accuracies = []
    learning_rates = []
    episode_lengths = []

    # Early stopping variables
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    patience_counter = 0
    best_model_path = models_dir / "dqn_model_best.pt"

    # Curriculum learning setup
    curriculum_episodes = []
    if curriculum_learning:
        stage_size = num_episodes // curriculum_stages
        for i in range(curriculum_stages):
            start_ep = i * stage_size
            end_ep = min((i + 1) * stage_size, num_episodes)
            curriculum_episodes.append((start_ep, end_ep, i + 1))
        logger.info(
            f"📚 Curriculum learning enabled with {curriculum_stages} stages")

    logger.info(
        f"🏋️  Starting enhanced training for {num_episodes} episodes...")
    logger.info(f"⏱️  Max steps per episode: {max_steps_per_episode}")
    logger.info(f"🧠 Warmup steps: {warmup_steps}")
    logger.info(f"🎯 Target update interval: {target_update_interval}")
    logger.info(f"🛡️  Gradient clipping: {grad_clip}")
    logger.info("=" * 70)

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        episode_start_time = time.time()

        # Determine current curriculum stage
        current_stage = 1
        stage_progress = 0.0
        if curriculum_learning:
            for start_ep, end_ep, stage in curriculum_episodes:
                if start_ep <= episode < end_ep:
                    current_stage = stage
                    stage_progress = (episode - start_ep) / (end_ep - start_ep)
                    break

        # === TRAINING PHASE ===
        state, info = train_env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        correct_predictions = 0
        total_predictions = 0
        step_count = 0

        # Adaptive episode length based on curriculum
        if curriculum_learning:
            adaptive_max_steps = int(
                max_steps_per_episode * (0.5 + 0.5 * stage_progress))
        else:
            adaptive_max_steps = max_steps_per_episode

        while not done and step_count < adaptive_max_steps:
            step_count += 1

            # Agent selects action
            action = agent.act(state=state, training=True)

            # Environment step
            next_state, reward, done, truncated, info = train_env.step(
                action=action)

            # Store experience in replay buffer
            agent.remember(state, action, reward,
                           next_state, done or truncated)

            # Train agent (after warmup)
            if len(agent.memory) >= warmup_steps:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

                    # Apply gradient clipping
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            agent.model.parameters(), grad_clip)

            # Track accuracy
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

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Calculate training metrics
        train_accuracy = correct_predictions / max(total_predictions, 1)
        avg_train_loss = np.mean(episode_losses) if episode_losses else 0.0
        current_lr = agent.optimizer.param_groups[0]['lr']
        episode_duration = time.time() - episode_start_time

        # Store training metrics
        train_rewards.append(total_reward)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        learning_rates.append(current_lr)
        episode_lengths.append(step_count)

        # Track best training accuracy
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        # === VALIDATION PHASE ===
        val_accuracy = 0.0
        val_reward = 0.0

        if use_validation and episode % validation_interval == 0:
            logger.debug(f"🔍 Running validation at episode {episode}")
            val_accuracy, val_reward = validate_agent(
                agent, val_env, adaptive_max_steps
            )
            val_accuracies.append(val_accuracy)
            val_rewards.append(val_reward)

            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                agent.save_model(best_model_path)
                logger.debug(
                    f"💾 New best model saved (val_acc: {val_accuracy:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered at episode {episode}")
                logger.info(
                    f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
                break

        # Enhanced logging
        if episode % 10 == 0 or episode == num_episodes - 1:
            log_msg = (
                f"Ep {episode + 1:4d}/{num_episodes} | "
                f"Stage {current_stage}/{curriculum_stages} | "
                f"Reward: {total_reward:7.2f} | "
                f"Acc: {train_accuracy:.4f} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Steps: {step_count:4d} | "
                f"Time: {episode_duration:.1f}s"
            )

            if use_validation and len(val_accuracies) > 0:
                log_msg += f" | Val: {val_accuracies[-1]:.4f}"

            logger.info(log_msg)

        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            model_path = episodes_dir / f"dqn_model_episode_{episode}.pt"
            agent.save_model(model_path)

    # Save final model
    final_model_path = models_dir / "dqn_model_final.pt"
    agent.save_model(final_model_path)
    logger.success(f"✅ Final model saved to: {final_model_path}")

    # Save enhanced training metrics
    metrics_data = {
        "Episode": range(1, len(train_rewards) + 1),
        "Train_Reward": train_rewards,
        "Train_Loss": train_losses,
        "Train_Accuracy": train_accuracies,
        "Learning_Rate": learning_rates,
        "Episode_Length": episode_lengths,
    }

    # Add validation metrics if available
    if use_validation and val_accuracies:
        # Extend validation metrics to match training episodes
        extended_val_acc = []
        extended_val_reward = []
        val_idx = 0

        for ep in range(len(train_rewards)):
            if ep % validation_interval == 0 and val_idx < len(val_accuracies):
                extended_val_acc.append(val_accuracies[val_idx])
                extended_val_reward.append(val_rewards[val_idx])
                val_idx += 1
            else:
                # Use last known validation value
                extended_val_acc.append(
                    extended_val_acc[-1] if extended_val_acc else 0.0)
                extended_val_reward.append(
                    extended_val_reward[-1] if extended_val_reward else 0.0)

        metrics_data["Val_Accuracy"] = extended_val_acc
        metrics_data["Val_Reward"] = extended_val_reward

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = reports_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.success(f"📈 Training metrics saved to: {metrics_path}")

    # Calculate and log enhanced statistics
    total_time = time.time() - start_time
    final_train_reward = np.mean(train_rewards[-20:])  # Last 20 episodes
    final_train_accuracy = np.mean(train_accuracies[-20:])
    max_train_accuracy = np.max(train_accuracies)
    avg_episode_length = np.mean(episode_lengths)

    # Training stability metrics
    recent_rewards = train_rewards[-50:] if len(
        train_rewards) >= 50 else train_rewards
    reward_stability = np.std(recent_rewards) / \
        (np.mean(recent_rewards) + 1e-8)

    recent_accuracies = train_accuracies[-50:] if len(
        train_accuracies) >= 50 else train_accuracies
    accuracy_stability = np.std(recent_accuracies)

    logger.info("\n" + "🎉 ENHANCED TRAINING COMPLETED!" + "\n" + "=" * 70)
    logger.info(f"⏱️  Total training time: {total_time / 60:.1f} minutes")
    logger.info(f"📊 Training Episodes: {len(train_rewards)}")
    logger.info(
        f"🏆 Final average train reward (last 20): {final_train_reward:.2f}")
    logger.info(
        f"🎯 Final average train accuracy (last 20): {final_train_accuracy:.4f}")
    logger.info(f"📈 Maximum train accuracy: {max_train_accuracy:.4f}")
    logger.info(
        f"📊 Best training accuracy achieved: {best_train_accuracy:.4f}")
    logger.info(f"⚡ Average episode length: {avg_episode_length:.1f} steps")
    logger.info(f"📊 Reward stability (CV): {reward_stability:.4f}")
    logger.info(f"📊 Accuracy stability (std): {accuracy_stability:.4f}")

    if use_validation and val_accuracies:
        logger.info(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info(f"💾 Best model saved to: {best_model_path}")

    logger.info(f"🔧 Final learning rate: {learning_rates[-1]:.2e}")
    logger.info(f"🎯 Final epsilon: {agent.epsilon:.4f}")
    logger.info(f"📁 Models directory: {models_dir}")
    logger.info(f"📋 Reports directory: {reports_dir}")
    logger.info("=" * 70)

    return {
        "final_train_reward": final_train_reward,
        "final_train_accuracy": final_train_accuracy,
        "max_train_accuracy": max_train_accuracy,
        "best_train_accuracy": best_train_accuracy,
        "best_val_accuracy": best_val_accuracy if use_validation else None,
        "total_episodes": len(train_rewards),
        "total_time_minutes": total_time / 60,
        "reward_stability": reward_stability,
        "accuracy_stability": accuracy_stability,
        "avg_episode_length": avg_episode_length,
        "model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if use_validation else None,
        "metrics_path": str(metrics_path),
    }


def validate_agent(agent, val_env, max_steps):
    """Run validation episode with trained agent"""
    # Store current training state
    training_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure greedy for validation

    state, info = val_env.reset()
    done = False
    total_reward = 0
    correct_predictions = 0
    total_predictions = 0
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1

        # Pure greedy action selection
        action = agent.act(state=state, training=False)

        # Environment step
        next_state, reward, done, truncated, info = val_env.step(action=action)

        # Track accuracy
        actual_label = info.get("actual_label", -1)
        if actual_label != -1:
            total_predictions += 1
            if action == actual_label:
                correct_predictions += 1

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Restore training epsilon
    agent.epsilon = training_epsilon

    accuracy = correct_predictions / max(total_predictions, 1)
    return accuracy, total_reward


if __name__ == "__main__":
    app()

# Advanced Training Tutorial

## Overview

This tutorial covers advanced training techniques for the RL-IDS system, including hyperparameter optimization, curriculum learning, transfer learning, and multi-objective training strategies.

## Prerequisites

- Completed [Getting Started Guide](../getting-started.md)
- Basic understanding of reinforcement learning concepts
- Familiarity with the RL-IDS environment and agent

## Learning Objectives

By the end of this tutorial, you will:

1. Master advanced DQN training techniques
2. Implement curriculum learning for improved convergence
3. Optimize hyperparameters systematically
4. Set up multi-stage training pipelines
5. Monitor and debug training issues

## 1. Advanced DQN Configuration

### Custom Agent Configuration

Start with a comprehensive DQN configuration:

```python
from rl_ids.agents.dqn_agent import DQNConfig, DQNAgent
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.config import TRAIN_DATA_FILE

# Advanced DQN configuration
config = DQNConfig(
    state_dim=77,              # Feature dimension
    action_dim=15,             # Number of attack classes
    
    # Network architecture
    hidden_dims=[512, 256, 128, 64],
    activation='relu',
    dropout_rate=0.1,
    
    # Learning parameters
    learning_rate=1e-4,
    lr_scheduler='cosine',      # Learning rate scheduling
    lr_patience=10,
    lr_factor=0.5,
    
    # Experience replay
    memory_size=100000,
    batch_size=64,
    min_memory_size=1000,
    
    # Exploration strategy
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    epsilon_decay_schedule='exponential',  # 'linear' or 'exponential'
    
    # Target network updates
    target_update_frequency=100,
    soft_target_update=True,
    tau=0.005,                  # Soft update parameter
    
    # Advanced features
    double_dqn=True,
    dueling_dqn=True,
    prioritized_replay=True,
    noisy_networks=False,
    
    # Training stability
    gradient_clipping=1.0,
    reward_scaling=1.0,
    
    # Device settings
    device='auto'  # 'cpu', 'cuda', or 'auto'
)

print(f"Configuration: {config}")
```

### Environment Configuration

Configure the environment with advanced features:

```python
# Advanced environment setup
env = IDSDetectionEnv(
    data_path=TRAIN_DATA_FILE,
    feature_cols=None,  # Auto-detect
    
    # Reward shaping
    reward_shaping='adaptive',     # 'fixed', 'adaptive', 'curriculum'
    attack_penalty=-1.0,
    benign_reward=0.1,
    correct_reward=1.0,
    wrong_penalty=-0.5,
    
    # Episode configuration
    max_episode_steps=1000,
    episode_termination='confidence',  # 'fixed', 'confidence', 'adaptive'
    confidence_threshold=0.95,
    
    # Data sampling
    sampling_strategy='balanced',      # 'sequential', 'random', 'balanced'
    class_weights='auto',
    
    # Difficulty progression
    curriculum_mode=True,
    difficulty_ramp=0.1,
    
    # Observation space
    observation_normalization=True,
    add_noise=False,
    noise_std=0.01
)

print(f"Environment configured with {env.observation_space.shape[0]} features")
```

## 2. Curriculum Learning Implementation

### Progressive Difficulty Training

Implement curriculum learning to improve training stability:

```python
import numpy as np
from typing import Dict, List
from loguru import logger

class CurriculumTrainer:
    """Implements curriculum learning for RL-IDS training"""
    
    def __init__(self, agent: DQNAgent, env: IDSDetectionEnv):
        self.agent = agent
        self.env = env
        self.curriculum_stages = []
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_performance = []
        
    def define_curriculum(self):
        """Define curriculum stages with increasing difficulty"""
        
        self.curriculum_stages = [
            {
                'name': 'Basic Binary Classification',
                'classes': ['BENIGN', 'DDoS'],
                'episodes': 200,
                'success_threshold': 0.85,
                'reward_multiplier': 1.0
            },
            {
                'name': 'Common Attack Types',
                'classes': ['BENIGN', 'DDoS', 'PortScan', 'Bot'],
                'episodes': 300,
                'success_threshold': 0.80,
                'reward_multiplier': 1.2
            },
            {
                'name': 'Web Attack Classification',
                'classes': ['BENIGN', 'Web Attack – Brute Force', 
                           'Web Attack – XSS', 'Web Attack – Sql Injection'],
                'episodes': 250,
                'success_threshold': 0.75,
                'reward_multiplier': 1.5
            },
            {
                'name': 'Advanced Infiltration',
                'classes': ['BENIGN', 'Infiltration', 'Heartbleed'],
                'episodes': 200,
                'success_threshold': 0.70,
                'reward_multiplier': 2.0
            },
            {
                'name': 'Full Multi-class',
                'classes': 'all',  # All 15 classes
                'episodes': 500,
                'success_threshold': 0.65,
                'reward_multiplier': 1.0
            }
        ]
        
        logger.info(f"Curriculum defined with {len(self.curriculum_stages)} stages")
    
    def setup_stage(self, stage_idx: int):
        """Setup environment for specific curriculum stage"""
        
        if stage_idx >= len(self.curriculum_stages):
            logger.info("Curriculum completed!")
            return False
            
        stage = self.curriculum_stages[stage_idx]
        logger.info(f"Starting curriculum stage {stage_idx + 1}: {stage['name']}")
        
        # Configure environment for this stage
        if stage['classes'] == 'all':
            self.env.set_active_classes(None)  # All classes
        else:
            self.env.set_active_classes(stage['classes'])
        
        # Adjust reward scaling
        self.env.reward_multiplier = stage['reward_multiplier']
        
        # Reset stage tracking
        self.stage_episodes = 0
        self.stage_performance = []
        
        return True
    
    def train_stage(self, stage_idx: int) -> bool:
        """Train on a specific curriculum stage"""
        
        stage = self.curriculum_stages[stage_idx]
        target_episodes = stage['episodes']
        success_threshold = stage['success_threshold']
        
        stage_metrics = {
            'episode_rewards': [],
            'episode_accuracies': [],
            'stage_losses': []
        }
        
        for episode in range(target_episodes):
            # Standard training episode
            episode_reward, episode_accuracy, loss = self.train_episode()
            
            # Track stage metrics
            stage_metrics['episode_rewards'].append(episode_reward)
            stage_metrics['episode_accuracies'].append(episode_accuracy)
            if loss is not None:
                stage_metrics['stage_losses'].append(loss)
            
            # Check stage completion criteria
            if episode >= 50:  # Minimum episodes before evaluation
                recent_accuracy = np.mean(stage_metrics['episode_accuracies'][-20:])
                
                if recent_accuracy >= success_threshold:
                    logger.success(
                        f"Stage {stage_idx + 1} completed! "
                        f"Accuracy: {recent_accuracy:.3f} >= {success_threshold:.3f}"
                    )
                    return True
            
            # Progress logging
            if episode % 50 == 0:
                avg_reward = np.mean(stage_metrics['episode_rewards'][-50:])
                avg_accuracy = np.mean(stage_metrics['episode_accuracies'][-50:])
                logger.info(
                    f"Stage {stage_idx + 1}, Episode {episode}: "
                    f"Avg Reward: {avg_reward:.3f}, Avg Accuracy: {avg_accuracy:.3f}"
                )
        
        # Stage completed by episode limit
        final_accuracy = np.mean(stage_metrics['episode_accuracies'][-20:])
        if final_accuracy >= success_threshold:
            logger.success(
                f"Stage {stage_idx + 1} completed by episode limit! "
                f"Final accuracy: {final_accuracy:.3f}"
            )
            return True
        else:
            logger.warning(
                f"Stage {stage_idx + 1} completed with suboptimal performance: "
                f"{final_accuracy:.3f} < {success_threshold:.3f}"
            )
            return False
    
    def train_episode(self) -> tuple:
        """Train a single episode and return metrics"""
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        correct_predictions = 0
        total_predictions = 0
        loss = None
        
        while True:
            # Agent action
            action = self.agent.act(state, training=True)
            
            # Environment step
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.memory) > self.agent.config.batch_size:
                loss = self.agent.replay()
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            
            # Track accuracy
            if 'correct' in info:
                total_predictions += 1
                if info['correct']:
                    correct_predictions += 1
            
            if done or truncated:
                break
                
            state = next_state
        
        # Calculate episode accuracy
        episode_accuracy = correct_predictions / max(total_predictions, 1)
        
        return episode_reward, episode_accuracy, loss
    
    def run_curriculum(self):
        """Execute the complete curriculum training"""
        
        self.define_curriculum()
        
        for stage_idx in range(len(self.curriculum_stages)):
            if not self.setup_stage(stage_idx):
                break
                
            success = self.train_stage(stage_idx)
            
            # Save stage checkpoint
            stage_name = self.curriculum_stages[stage_idx]['name'].replace(' ', '_')
            checkpoint_path = f"models/dqn_curriculum_stage_{stage_idx + 1}_{stage_name}.pt"
            self.agent.save_model(checkpoint_path)
            
            if not success:
                logger.warning(f"Consider extending training for stage {stage_idx + 1}")
        
        logger.success("Curriculum training completed!")

# Run curriculum training
curriculum_trainer = CurriculumTrainer(agent, env)
curriculum_trainer.run_curriculum()
```

## 3. Hyperparameter Optimization

### Automated Hyperparameter Search

Implement systematic hyperparameter optimization:

```python
import itertools
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

class HyperparameterOptimizer:
    """Systematic hyperparameter optimization for DQN agent"""
    
    def __init__(self, base_config: DQNConfig):
        self.base_config = base_config
        self.search_space = {}
        self.results = []
        
    def define_search_space(self):
        """Define hyperparameter search space"""
        
        self.search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64, 128, 256],
            'memory_size': [50000, 100000, 200000],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'target_update_frequency': [50, 100, 200],
            'hidden_dims': [
                [256, 128],
                [512, 256, 128],
                [512, 256, 128, 64],
                [1024, 512, 256, 128]
            ],
            'double_dqn': [True, False],
            'dueling_dqn': [True, False],
            'prioritized_replay': [True, False]
        }
    
    def grid_search(self, max_combinations: int = 50) -> List[Dict]:
        """Perform grid search over hyperparameter space"""
        
        self.define_search_space()
        
        # Generate all combinations
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        
        all_combinations = list(itertools.product(*values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            # Random sampling of combinations
            import random
            combinations = random.sample(all_combinations, max_combinations)
            logger.info(f"Randomly sampling {max_combinations} from {len(all_combinations)} combinations")
        else:
            combinations = all_combinations
            logger.info(f"Testing all {len(combinations)} combinations")
        
        results = []
        
        for i, combination in enumerate(combinations):
            logger.info(f"Testing combination {i + 1}/{len(combinations)}")
            
            # Create configuration
            config_dict = dict(zip(keys, combination))
            config = self.create_config(config_dict)
            
            # Train and evaluate
            performance = self.evaluate_configuration(config, trial_episodes=100)
            
            # Store results
            result = {
                'config': config_dict,
                'performance': performance,
                'trial': i + 1
            }
            results.append(result)
            
            # Log intermediate result
            logger.info(
                f"Trial {i + 1} - Accuracy: {performance['accuracy']:.3f}, "
                f"Reward: {performance['avg_reward']:.3f}"
            )
            
            # Save intermediate results
            self.save_results(results, f"hyperopt_intermediate_{i + 1}.json")
        
        # Sort by performance
        results.sort(key=lambda x: x['performance']['accuracy'], reverse=True)
        
        # Save final results
        self.save_results(results, "hyperopt_final_results.json")
        
        return results
    
    def create_config(self, config_dict: Dict[str, Any]) -> DQNConfig:
        """Create DQNConfig from hyperparameter dictionary"""
        
        # Start with base config
        config = DQNConfig(
            state_dim=self.base_config.state_dim,
            action_dim=self.base_config.action_dim
        )
        
        # Update with search parameters
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        return config
    
    def evaluate_configuration(self, config: DQNConfig, trial_episodes: int = 100) -> Dict[str, float]:
        """Evaluate a configuration with limited training"""
        
        # Create agent and environment
        agent = DQNAgent(config)
        env = IDSDetectionEnv(data_path=TRAIN_DATA_FILE)
        
        episode_rewards = []
        episode_accuracies = []
        
        for episode in range(trial_episodes):
            state, _ = env.reset()
            episode_reward = 0
            correct_predictions = 0
            total_predictions = 0
            
            while True:
                action = agent.act(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > config.batch_size:
                    agent.replay()
                
                episode_reward += reward
                
                if 'correct' in info:
                    total_predictions += 1
                    if info['correct']:
                        correct_predictions += 1
                
                if done or truncated:
                    break
                    
                state = next_state
            
            episode_accuracy = correct_predictions / max(total_predictions, 1)
            episode_rewards.append(episode_reward)
            episode_accuracies.append(episode_accuracy)
        
        # Calculate performance metrics
        performance = {
            'accuracy': np.mean(episode_accuracies),
            'accuracy_std': np.std(episode_accuracies),
            'avg_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'final_epsilon': agent.epsilon,
            'convergence_episodes': self.detect_convergence(episode_accuracies)
        }
        
        return performance
    
    def detect_convergence(self, accuracies: List[float], window: int = 20) -> int:
        """Detect convergence point in training"""
        
        if len(accuracies) < window * 2:
            return len(accuracies)
        
        for i in range(window, len(accuracies) - window):
            early_mean = np.mean(accuracies[i-window:i])
            late_mean = np.mean(accuracies[i:i+window])
            
            # Check if improvement is minimal
            if abs(late_mean - early_mean) < 0.01:
                return i
        
        return len(accuracies)
    
    def save_results(self, results: List[Dict], filename: str):
        """Save optimization results to JSON"""
        
        results_path = Path("hyperopt_results") / filename
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def get_best_configuration(self, results: List[Dict]) -> DQNConfig:
        """Get the best performing configuration"""
        
        best_result = max(results, key=lambda x: x['performance']['accuracy'])
        best_config = self.create_config(best_result['config'])
        
        logger.success(
            f"Best configuration - Accuracy: {best_result['performance']['accuracy']:.3f}"
        )
        
        return best_config

# Run hyperparameter optimization
base_config = DQNConfig(state_dim=77, action_dim=15)
optimizer = HyperparameterOptimizer(base_config)

# Perform optimization
results = optimizer.grid_search(max_combinations=30)

# Get best configuration
best_config = optimizer.get_best_configuration(results)
```

## 4. Multi-Objective Training

### Balancing Multiple Objectives

Implement multi-objective training for improved performance:

```python
from typing import Dict, List, Tuple
import numpy as np

class MultiObjectiveTrainer:
    """Multi-objective training for RL-IDS with balanced performance"""
    
    def __init__(self, agent: DQNAgent, env: IDSDetectionEnv):
        self.agent = agent
        self.env = env
        self.objectives = {}
        self.objective_weights = {}
        self.performance_history = []
        
    def define_objectives(self):
        """Define training objectives and their weights"""
        
        self.objectives = {
            'accuracy': {
                'description': 'Overall classification accuracy',
                'target': 0.90,
                'weight': 0.4,
                'maximize': True
            },
            'precision': {
                'description': 'Precision for attack detection',
                'target': 0.85,
                'weight': 0.3,
                'maximize': True
            },
            'recall': {
                'description': 'Recall for attack detection',
                'target': 0.85,
                'weight': 0.2,
                'maximize': True
            },
            'false_positive_rate': {
                'description': 'False positive rate for benign traffic',
                'target': 0.05,
                'weight': 0.1,
                'maximize': False
            }
        }
        
        logger.info(f"Defined {len(self.objectives)} training objectives")
    
    def compute_multi_objective_reward(self, metrics: Dict[str, float]) -> float:
        """Compute combined reward from multiple objectives"""
        
        total_reward = 0.0
        objective_rewards = {}
        
        for obj_name, obj_config in self.objectives.items():
            if obj_name in metrics:
                value = metrics[obj_name]
                target = obj_config['target']
                weight = obj_config['weight']
                maximize = obj_config['maximize']
                
                # Compute objective-specific reward
                if maximize:
                    # Reward for exceeding target
                    obj_reward = max(0, (value - target) / target)
                else:
                    # Penalty for exceeding target (for metrics like FPR)
                    obj_reward = max(0, (target - value) / target)
                
                weighted_reward = obj_reward * weight
                objective_rewards[obj_name] = weighted_reward
                total_reward += weighted_reward
        
        return total_reward, objective_rewards
    
    def train_episode_multi_objective(self) -> Dict[str, float]:
        """Train episode with multi-objective evaluation"""
        
        state, _ = self.env.reset()
        episode_metrics = {
            'reward': 0,
            'steps': 0,
            'correct_predictions': 0,
            'total_predictions': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        while True:
            action = self.agent.act(state, training=True)
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Standard RL update
            self.agent.remember(state, action, reward, next_state, done)
            
            if len(self.agent.memory) > self.agent.config.batch_size:
                self.agent.replay()
            
            # Collect detailed metrics
            episode_metrics['reward'] += reward
            episode_metrics['steps'] += 1
            
            if 'true_label' in info and 'predicted_label' in info:
                true_label = info['true_label']
                pred_label = info['predicted_label']
                
                episode_metrics['total_predictions'] += 1
                
                # Binary classification metrics (attack vs benign)
                is_attack_true = true_label != 0  # Assuming 0 is benign
                is_attack_pred = pred_label != 0
                
                if is_attack_true and is_attack_pred:
                    episode_metrics['true_positives'] += 1
                    episode_metrics['correct_predictions'] += 1
                elif not is_attack_true and not is_attack_pred:
                    episode_metrics['true_negatives'] += 1
                    episode_metrics['correct_predictions'] += 1
                elif not is_attack_true and is_attack_pred:
                    episode_metrics['false_positives'] += 1
                else:  # is_attack_true and not is_attack_pred
                    episode_metrics['false_negatives'] += 1
            
            if done or truncated:
                break
                
            state = next_state
        
        # Compute derived metrics
        tp = episode_metrics['true_positives']
        fp = episode_metrics['false_positives']
        tn = episode_metrics['true_negatives']
        fn = episode_metrics['false_negatives']
        
        # Calculate objective metrics
        accuracy = episode_metrics['correct_predictions'] / max(episode_metrics['total_predictions'], 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        
        performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'false_positive_rate': fpr,
            'episode_reward': episode_metrics['reward'],
            'episode_steps': episode_metrics['steps']
        }
        
        # Compute multi-objective reward
        mo_reward, obj_rewards = self.compute_multi_objective_reward(performance_metrics)
        performance_metrics['multi_objective_reward'] = mo_reward
        performance_metrics['objective_rewards'] = obj_rewards
        
        return performance_metrics
    
    def train_multi_objective(self, episodes: int = 1000):
        """Main multi-objective training loop"""
        
        self.define_objectives()
        episode_history = []
        
        for episode in range(episodes):
            # Train episode
            metrics = self.train_episode_multi_objective()
            episode_history.append(metrics)
            
            # Adaptive objective weighting
            if episode > 50 and episode % 100 == 0:
                self.adapt_objective_weights(episode_history[-100:])
            
            # Progress logging
            if episode % 50 == 0:
                recent_metrics = episode_history[-50:] if len(episode_history) >= 50 else episode_history
                avg_metrics = self.compute_average_metrics(recent_metrics)
                
                logger.info(
                    f"Episode {episode}: "
                    f"Acc: {avg_metrics['accuracy']:.3f}, "
                    f"Prec: {avg_metrics['precision']:.3f}, "
                    f"Rec: {avg_metrics['recall']:.3f}, "
                    f"FPR: {avg_metrics['false_positive_rate']:.3f}, "
                    f"MO Reward: {avg_metrics['multi_objective_reward']:.3f}"
                )
        
        self.performance_history = episode_history
        return episode_history
    
    def adapt_objective_weights(self, recent_history: List[Dict[str, float]]):
        """Adaptively adjust objective weights based on performance"""
        
        avg_metrics = self.compute_average_metrics(recent_history)
        
        for obj_name, obj_config in self.objectives.items():
            if obj_name in avg_metrics:
                current_value = avg_metrics[obj_name]
                target_value = obj_config['target']
                current_weight = obj_config['weight']
                
                # Increase weight for underperforming objectives
                if obj_config['maximize']:
                    if current_value < target_value * 0.9:  # Performing below 90% of target
                        obj_config['weight'] = min(current_weight * 1.1, 0.5)
                else:
                    if current_value > target_value * 1.1:  # Performing above 110% of target
                        obj_config['weight'] = min(current_weight * 1.1, 0.5)
        
        # Normalize weights
        total_weight = sum(obj['weight'] for obj in self.objectives.values())
        for obj_config in self.objectives.values():
            obj_config['weight'] /= total_weight
        
        logger.debug(f"Updated objective weights: {[(k, v['weight']) for k, v in self.objectives.items()]}")
    
    def compute_average_metrics(self, history: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average metrics over episode history"""
        
        if not history:
            return {}
        
        metrics = {}
        for key in history[0].keys():
            if key != 'objective_rewards' and isinstance(history[0][key], (int, float)):
                metrics[key] = np.mean([episode[key] for episode in history])
        
        return metrics

# Run multi-objective training
mo_trainer = MultiObjectiveTrainer(agent, env)
training_history = mo_trainer.train_multi_objective(episodes=1000)
```

## 5. Advanced Monitoring and Debugging

### Comprehensive Training Monitoring

Set up advanced monitoring for training diagnostics:

```python
import wandb
from tensorboard import program
import matplotlib.pyplot as plt
from typing import Dict, Any

class AdvancedTrainingMonitor:
    """Comprehensive training monitoring with multiple backends"""
    
    def __init__(self, project_name: str = "rl-ids-advanced"):
        self.project_name = project_name
        self.metrics_history = []
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """Setup monitoring backends"""
        
        # Weights & Biases setup
        try:
            wandb.init(project=self.project_name, reinit=True)
            self.use_wandb = True
            logger.info("W&B monitoring enabled")
        except Exception as e:
            logger.warning(f"W&B setup failed: {e}")
            self.use_wandb = False
        
        # TensorBoard setup
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=f"runs/{self.project_name}")
            self.use_tensorboard = True
            logger.info("TensorBoard monitoring enabled")
        except Exception as e:
            logger.warning(f"TensorBoard setup failed: {e}")
            self.use_tensorboard = False
    
    def log_training_step(self, episode: int, metrics: Dict[str, Any]):
        """Log training metrics to all monitoring backends"""
        
        # Store locally
        metrics['episode'] = episode
        self.metrics_history.append(metrics.copy())
        
        # W&B logging
        if self.use_wandb:
            try:
                wandb.log(metrics, step=episode)
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")
        
        # TensorBoard logging
        if self.use_tensorboard:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(key, value, episode)
            except Exception as e:
                logger.warning(f"TensorBoard logging failed: {e}")
    
    def log_network_analysis(self, agent: DQNAgent, episode: int):
        """Log detailed network analysis"""
        
        # Gradient analysis
        total_norm = 0
        param_count = 0
        for param in agent.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += param.numel()
        
        total_norm = total_norm ** (1. / 2)
        
        # Weight analysis
        weight_stats = {}
        for name, param in agent.model.named_parameters():
            weight_stats[f'weight_mean_{name}'] = param.data.mean().item()
            weight_stats[f'weight_std_{name}'] = param.data.std().item()
            weight_stats[f'weight_norm_{name}'] = param.data.norm().item()
        
        # Q-value analysis
        if hasattr(agent, 'last_q_values') and agent.last_q_values is not None:
            q_values = agent.last_q_values.detach().cpu().numpy()
            q_stats = {
                'q_values_mean': np.mean(q_values),
                'q_values_std': np.std(q_values),
                'q_values_max': np.max(q_values),
                'q_values_min': np.min(q_values)
            }
        else:
            q_stats = {}
        
        analysis_metrics = {
            'gradient_norm': total_norm,
            'parameter_count': param_count,
            'learning_rate': agent.optimizer.param_groups[0]['lr'],
            'epsilon': agent.epsilon,
            **weight_stats,
            **q_stats
        }
        
        self.log_training_step(episode, analysis_metrics)
    
    def log_hyperparameters(self, config: DQNConfig):
        """Log hyperparameters for experiment tracking"""
        
        config_dict = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'memory_size': config.memory_size,
            'epsilon_decay': config.epsilon_decay,
            'target_update_frequency': config.target_update_frequency,
            'hidden_dims': str(config.hidden_dims),
            'double_dqn': config.double_dqn,
            'dueling_dqn': config.dueling_dqn,
            'prioritized_replay': config.prioritized_replay
        }
        
        if self.use_wandb:
            wandb.config.update(config_dict)
        
        logger.info(f"Logged hyperparameters: {config_dict}")
    
    def generate_training_report(self, save_path: str = "training_report.html"):
        """Generate comprehensive training report"""
        
        if not self.metrics_history:
            logger.warning("No metrics history available for report")
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Advanced Training Analysis Report', fontsize=16)
        
        # Accuracy trend
        if 'accuracy' in df.columns:
            axes[0, 0].plot(df['episode'], df['accuracy'])
            axes[0, 0].set_title('Training Accuracy')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Accuracy')
        
        # Reward trend
        if 'episode_reward' in df.columns:
            axes[0, 1].plot(df['episode'], df['episode_reward'])
            axes[0, 1].set_title('Episode Reward')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
        
        # Loss trend
        if 'loss' in df.columns:
            axes[1, 0].plot(df['episode'], df['loss'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        if 'epsilon' in df.columns:
            axes[1, 1].plot(df['episode'], df['epsilon'])
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
        
        # Gradient norm
        if 'gradient_norm' in df.columns:
            axes[2, 0].plot(df['episode'], df['gradient_norm'])
            axes[2, 0].set_title('Gradient Norm')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Gradient Norm')
        
        # Learning rate
        if 'learning_rate' in df.columns:
            axes[2, 1].plot(df['episode'], df['learning_rate'])
            axes[2, 1].set_title('Learning Rate')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL-IDS Advanced Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .plot {{ text-align: center; margin: 30px 0; }}
                h1, h2 {{ color: #2E86AB; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>RL-IDS Advanced Training Report</h1>
            
            <h2>Training Summary</h2>
            <table>
                <tr><th>Metric</th><th>Final Value</th><th>Best Value</th><th>Mean Value</th></tr>
                <tr><td>Accuracy</td><td>{df['accuracy'].iloc[-1]:.4f}</td><td>{df['accuracy'].max():.4f}</td><td>{df['accuracy'].mean():.4f}</td></tr>
                <tr><td>Reward</td><td>{df['episode_reward'].iloc[-1]:.2f}</td><td>{df['episode_reward'].max():.2f}</td><td>{df['episode_reward'].mean():.2f}</td></tr>
                <tr><td>Episodes</td><td>{len(df)}</td><td>-</td><td>-</td></tr>
            </table>
            
            <div class="plot">
                <img src="{save_path.replace('.html', '.png')}" alt="Training Analysis Plots" style="max-width: 100%;">
            </div>
            
            <h2>Performance Analysis</h2>
            <p>The training completed {len(df)} episodes with final accuracy of {df['accuracy'].iloc[-1]:.4f}.</p>
            <p>Best performance achieved: {df['accuracy'].max():.4f} accuracy at episode {df.loc[df['accuracy'].idxmax(), 'episode']}.</p>
            
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.success(f"Training report generated: {save_path}")

# Setup advanced monitoring
monitor = AdvancedTrainingMonitor("rl-ids-advanced-training")
monitor.log_hyperparameters(config)

# Enhanced training loop with monitoring
for episode in range(1000):
    # Train episode
    metrics = train_episode()  # Your training function
    
    # Log basic metrics
    monitor.log_training_step(episode, metrics)
    
    # Detailed network analysis every 10 episodes
    if episode % 10 == 0:
        monitor.log_network_analysis(agent, episode)
    
    # Generate intermediate reports
    if episode % 200 == 0 and episode > 0:
        monitor.generate_training_report(f"training_report_episode_{episode}.html")

# Final report
monitor.generate_training_report("final_training_report.html")
```

## 6. Next Steps

After completing this advanced training tutorial:

1. **Experiment with different curriculum strategies**
2. **Implement custom reward shaping techniques**
3. **Explore ensemble methods and model averaging**
4. **Set up automated hyperparameter optimization pipelines**
5. **Deploy trained models to production environments**

## See Also

- [Model Evaluation Tutorial](evaluation.md) - Comprehensive model evaluation
- [API Deployment Tutorial](deployment.md) - Production deployment strategies
- [Custom Environment Tutorial](custom_environments.md) - Creating custom training environments
- [Performance Optimization Guide](optimization.md) - Advanced performance tuning

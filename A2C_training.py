#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) Training for PlantOS Environment

This module implements A2C training using Stable Baselines3 for the PlantOS
exploration and plant-watering task. A2C is an on-policy algorithm that can
work well for this task with proper hyperparameter tuning.
"""

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from plantos_env import PlantOSEnv


# Create directories for logs and models
log_dir = "a2c_training/logs/"
models_dir = "a2c_training/models/"
videos_dir = "a2c_training/videos/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)




class CurriculumWrapper(gym.Wrapper):
    """
    PROGRESSIVE curriculum: Start with low threshold, gradually increase.
    """
    def __init__(self, env, initial_threshold=40.0, max_threshold=100.0, threshold_increment=10.0):
        super().__init__(env)
        self.maze_completed = False
        self.episode_count = 0
        self.successful_explorations = 0
        self.current_maze_seed = None
        self.persistent_visit_counts = None
        

        self.exploration_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.threshold_increment = threshold_increment
        self.episodes_on_current_maze = 0
        self.max_episodes_per_maze = 3  # Quick maze changes for A2C
        
    def reset(self, **kwargs):
        """Only generate new maze if threshold reached OR max attempts exceeded."""
        self.episode_count += 1
        self.episodes_on_current_maze += 1
        

        kwargs.pop('seed', None)
        

        timeout = self.episodes_on_current_maze >= self.max_episodes_per_maze
        
        if self.maze_completed or timeout:
            if self.maze_completed:
                self.exploration_threshold = min(
                    self.exploration_threshold + self.threshold_increment,
                    self.max_threshold
                )
                self.successful_explorations += 1
            
            self.maze_completed = False
            self.episodes_on_current_maze = 0
            

            self.current_maze_seed = np.random.randint(0, 10000)
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
            self.persistent_visit_counts = None
        else:

            if self.current_maze_seed is None:
                self.current_maze_seed = np.random.randint(0, 10000)
            
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
            

            if self.persistent_visit_counts is not None:
                self.env.visit_counts = self.persistent_visit_counts.copy()
            else:
                self.persistent_visit_counts = self.env.visit_counts.copy()
            
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        

        if info['exploration_percentage'] >= self.exploration_threshold:
            self.maze_completed = True
            terminated = True
        

        if self.persistent_visit_counts is not None:
            self.persistent_visit_counts = self.env.visit_counts.copy()
            
        return obs, reward, terminated, truncated, info




def make_env_wrapper(env_kwargs, rank=0, use_curriculum=True):
    """Helper function to create environment factory for vectorized training."""
    def _init():
        env = PlantOSEnv(**env_kwargs)
        

        if use_curriculum:
            env = CurriculumWrapper(env, initial_threshold=40.0, max_threshold=100.0)
        

        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init




class SaveOnIntervalCallback(BaseCallback):
    """Callback to save model at specific intervals."""
    def __init__(self, save_interval: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f'a2c_model_{self.num_timesteps}')
            self.model.save(save_file)
            if self.verbose > 0:
                print(f'💾 Saving model to {save_file}.zip')
        return True


class EvaluationCallback(BaseCallback):
    """Custom callback for logging exploration progress."""
    
    def __init__(self, log_dir, eval_freq=10000):
        super().__init__()
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.best_mean_exploration = 0
        self.exploration_history = []
        self.maze_completion_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:

            if len(self.model.ep_info_buffer) > 0:
                recent_episodes = list(self.model.ep_info_buffer)[-10:]
                

                explorations = []
                for ep_info in recent_episodes:
                    if 'exploration_percentage' in ep_info:
                        explorations.append(ep_info['exploration_percentage'])
                        if ep_info['exploration_percentage'] >= 100.0:
                            self.maze_completion_count += 1
                
                if explorations:
                    mean_exploration = np.mean(explorations)
                    self.exploration_history.append(mean_exploration)
                    

                    with open(f"{self.log_dir}training_log.txt", "a") as f:
                        f.write(f"[Step {self.n_calls}] Mean Exploration: {mean_exploration:.2f}%\n")
                        f.write(f"Mazes completed: {self.maze_completion_count}\n")
                    
                    if mean_exploration > self.best_mean_exploration:
                        self.best_mean_exploration = mean_exploration
        
        return True




def train_with_a2c(n_envs=8, use_curriculum=False, total_timesteps=100000): #100k
    """
    Train using A2C (Advantage Actor-Critic).
    
    A2C is a synchronous version of A3C that:
    - Uses multiple parallel environments
    - Is on-policy (like PPO)
    - Is faster than RecurrentPPO
    - Works well with continuous learning
    
    Args:
        n_envs: Number of parallel environments
        use_curriculum: Whether to use curriculum learning
        total_timesteps: Total training steps
    """
    

    env_kwargs = {
        'grid_size': 25,
        'num_plants': 10,
        'num_obstacles': 12,
        'lidar_range': 6,
        'lidar_channels': 16
    }
    

    print(f"Creating {n_envs} parallel environments...")
    env_fns = [make_env_wrapper(env_kwargs, rank=i, use_curriculum=use_curriculum) 
               for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    

    
    print("=" * 60)
    print(f"Training with A2C (Advantage Actor-Critic)")
    print(f"Parallel Environments: {n_envs}")
    print(f"Curriculum Learning: {'ENABLED' if use_curriculum else 'DISABLED'}")
    print("=" * 60)
    

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,           # Learning rate (A2C typically uses higher LR than PPO)
        n_steps=5,                    # Steps per environment before update (A2C uses small values)
        gamma=0.99,                   # Discount factor
        gae_lambda=1.0,               # GAE lambda (1.0 = Monte Carlo, 0.0 = TD)
        ent_coef=0.01,                # Entropy coefficient for exploration
        vf_coef=0.25,                 # Value function coefficient
        max_grad_norm=0.5,            # Gradient clipping
        rms_prop_eps=1e-5,            # RMSProp epsilon
        use_rms_prop=True,            # Use RMSProp optimizer (default for A2C)
        normalize_advantage=True,      # Normalize advantages
        verbose=1,
        tensorboard_log=f"{log_dir}tensorboard/",
        policy_kwargs=dict(
            net_arch=[256, 256]      # Network architecture
        )
    )
    

    save_interval = total_timesteps // 10  # Save 10 times during training
    checkpoint_callback = SaveOnIntervalCallback(
        save_interval=save_interval,
        save_path=models_dir
    )
    
    eval_callback = EvaluationCallback(log_dir, eval_freq=10000)
    

    print("\nStarting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Checkpoints every: {save_interval:,} steps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    

    final_model_path = f"{models_dir}a2c_final"
    model.save(final_model_path)
    print(f"\n Training complete! Final model saved to: {final_model_path}")
    

    print("\n" + "=" * 60)
    print("Evaluating trained model...")
    print("=" * 60)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    

    plot_learning_curve(log_dir, "A2C Learning Curve")
    

    env.close()
    
    return model




def plot_learning_curve(log_dir, title="Learning Curve"):
    """Plot the learning curve from training logs."""
    try:

        results = load_results(log_dir)
        
        if len(results) == 0:
            print("No results to plot yet.")
            return
        

        results = results.sort_values('t')
        
        x, y = ts2xy(results, 'timesteps')
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        

        ax1.plot(x, y, alpha=0.3, color='blue', label='Raw Reward')
        

        if len(y) > 100:
            window = min(100, len(y) // 10)
            y_smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smoothed = x[:len(y_smoothed)]
            ax1.plot(x_smoothed, y_smoothed, color='red', linewidth=2, label='Smoothed Reward')
        
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        

        x_len, y_len = ts2xy(results, 'timesteps')
        ax2.plot(x_len, results['l'].values, alpha=0.3, color='green')
        
        if len(results['l'].values) > 100:
            window = min(100, len(results['l'].values) // 10)
            y_len_smoothed = np.convolve(results['l'].values, np.ones(window)/window, mode='valid')
            x_len_smoothed = x_len[:len(y_len_smoothed)]
            ax2.plot(x_len_smoothed, y_len_smoothed, color='orange', linewidth=2)
        
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}learning_curve.png", dpi=150)
        print(f"\n📊 Learning curve saved to: {log_dir}learning_curve.png")
        plt.show()
        
    except Exception as e:
        print(f"Error plotting learning curve: {e}")




def test_a2c_model(model_path, num_episodes=5):
    """Test a trained A2C model and visualize its performance."""
    

    env = PlantOSEnv(
        grid_size=25,
        num_plants=10,
        num_obstacles=12,
        lidar_range=6,
        lidar_channels=16,
        render_mode='2d'
    )
    

    try:
        model = A2C.load(model_path)
        print(f"Loaded A2C model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        while not done:

            action, _ = model.predict(obs, deterministic=True)
            

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            

            env.render('2d')
            time.sleep(0.01)
            

            if steps % 100 == 0:
                print(f"  Step {steps}: Exploration {info['exploration_percentage']:.1f}%, "
                      f"Reward: {total_reward:.1f}")
        

        print(f"\n Episode {episode + 1} Complete")
        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Exploration: {info['exploration_percentage']:.2f}%")
        print(f"   Thirsty Plants: {info['thirsty_plants']}/{info['total_plants']}")
        print(f"   Plants Watered: {info['total_plants'] - info['thirsty_plants']}/{info['total_plants']}")
    
    env.close()




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A2C Trainer for PlantOS Environment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--model-path', type=str, default='a2c_training/models/a2c_final',
                       help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n" + "="*60)
        print("A2C Training for PlantOS Environment")
        print("="*60)
        
        model = train_with_a2c(
            n_envs=args.envs,
            use_curriculum=args.curriculum,
            total_timesteps=args.timesteps
        )
        

        test = input("\nTest the trained model? (y/n): ").strip().lower()
        if test == 'y':
            test_a2c_model(f"{models_dir}a2c_final", num_episodes=args.episodes)
    
    elif args.mode == 'test':
        print("\n" + "="*60)
        print("Testing A2C Model")
        print("="*60)
        
        test_a2c_model(args.model_path, num_episodes=args.episodes)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

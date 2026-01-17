#!/usr/bin/env python3
"""
Monte Carlo Tree Search (MCTS) Trainer for PlantOS Environment

This module implements MCTS for the exploration task. MCTS is particularly
well-suited for planning and exploration as it can look ahead and evaluate
different action sequences before committing to a decision.
"""

import numpy as np
import math
import pickle
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from plantos_env import PlantOSEnv
import time


class MCTSNode:
    """
    Node in the MCTS tree.
    Each node represents a state in the environment.
    """
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state  # Observation
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}  # Dict: action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(range(5))  # 0-3: movement, 4: water
        
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param: float = 1.414) -> 'MCTSNode':
        """
        Select best child using UCB1 (Upper Confidence Bound) formula.
        
        Args:
            c_param: Exploration parameter (sqrt(2) is theoretical optimum)
        
        Returns:
            Best child node
        """
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')  # Prioritize unvisited nodes
            else:
                # UCB1: exploitation + exploration
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append((child, weight))
        
        return max(choices_weights, key=lambda x: x[1])[0]
    
    def best_action(self) -> int:
        """Get the action with highest average value (for final decision)."""
        if not self.children:
            return np.random.randint(5)
        
        best_child = max(self.children.values(), 
                        key=lambda node: node.value / max(node.visits, 1))
        return best_child.action


class MCTS:
    """
    Monte Carlo Tree Search implementation for PlantOS environment.
    """
    def __init__(self, env: PlantOSEnv, n_simulations: int = 100, 
                 c_param: float = 1.414, max_depth: int = 50):
        """
        Initialize MCTS.
        
        Args:
            env: PlantOS environment
            n_simulations: Number of MCTS simulations per action selection
            c_param: Exploration parameter for UCB1
            max_depth: Maximum depth for tree search/rollout
        """
        self.env = env
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.max_depth = max_depth
        
    def search(self, initial_state: np.ndarray) -> int:
        """
        Perform MCTS search from the initial state.
        
        Args:
            initial_state: Current observation from environment
        
        Returns:
            Best action to take
        """
        root = MCTSNode(initial_state)
        
        for _ in range(self.n_simulations):
            node = root
            
            # Create a copy of the environment for simulation
            sim_env = self._copy_env_state()
            
            # 1. SELECTION: Traverse tree using UCB1
            depth = 0
            while node.is_fully_expanded() and node.children and depth < self.max_depth:
                node = node.best_child(self.c_param)
                obs, reward, terminated, truncated, info = sim_env.step(node.action)
                depth += 1
                
                if terminated or truncated:
                    break
            
            # 2. EXPANSION: Add a new child node
            if not node.is_fully_expanded() and depth < self.max_depth:
                action = node.untried_actions.pop(np.random.randint(len(node.untried_actions)))
                obs, reward, terminated, truncated, info = sim_env.step(action)
                child_node = MCTSNode(obs, parent=node, action=action)
                node.children[action] = child_node
                node = child_node
            
            # 3. SIMULATION (Rollout): Play randomly until terminal state
            rollout_reward = self._rollout(sim_env, depth)
            
            # 4. BACKPROPAGATION: Update node values
            while node is not None:
                node.visits += 1
                node.value += rollout_reward
                node = node.parent
        
        # Return best action from root
        return root.best_action()
    
    def _rollout(self, env: PlantOSEnv, current_depth: int) -> float:
        """
        Perform random rollout from current state.
        
        Args:
            env: Environment to simulate in
            current_depth: Current depth in the tree
        
        Returns:
            Total reward from rollout
        """
        total_reward = 0.0
        depth = current_depth
        
        while depth < self.max_depth:
            # Use a simple heuristic policy for rollout
            action = self._rollout_policy(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            depth += 1
            
            if terminated or truncated:
                # Bonus for completing exploration
                if info.get('exploration_percentage', 0) >= 100:
                    total_reward += 500
                break
        
        return total_reward
    
    def _rollout_policy(self, env: PlantOSEnv) -> int:
        """
        Smart rollout policy that favors exploration.
        
        Args:
            env: Environment
        
        Returns:
            Action to take
        """
        # 70% chance to move to least visited adjacent cell
        # 30% chance to take random action
        if np.random.random() < 0.7:
            return self._exploration_heuristic(env)
        else:
            return np.random.randint(5)
    
    def _exploration_heuristic(self, env: PlantOSEnv) -> int:
        """
        Heuristic that prefers moving to less-visited cells.
        
        Args:
            env: Environment
        
        Returns:
            Action that leads to least visited cell
        """
        rover_x, rover_y = env.rover_pos
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # North, East, South, West
        
        best_action = None
        min_visits = float('inf')
        
        for action, (dx, dy) in enumerate(directions):
            new_x = rover_x + dx
            new_y = rover_y + dy
            
            # Check if valid move
            if (0 <= new_x < env.grid_size and 
                0 <= new_y < env.grid_size and 
                (new_x, new_y) not in env.obstacles):
                
                visits = env.visit_counts[new_x, new_y]
                if visits < min_visits:
                    min_visits = visits
                    best_action = action
        
        # If all moves are blocked, return random action
        return best_action if best_action is not None else np.random.randint(5)
    
    def _copy_env_state(self) -> PlantOSEnv:
        """
        Create a copy of the environment for simulation.
        This is a lightweight copy that preserves the current state.
        
        Returns:
            Copy of environment
        """
        # Create new environment with same parameters
        sim_env = PlantOSEnv(
            grid_size=self.env.grid_size,
            num_plants=self.env.num_plants,
            num_obstacles=self.env.num_obstacles,
            lidar_range=self.env.lidar_range,
            lidar_channels=self.env.lidar_channels
        )
        
        # Copy state
        sim_env.rover_pos = self.env.rover_pos
        sim_env.plants = self.env.plants.copy()
        sim_env.obstacles = self.env.obstacles.copy()
        sim_env.explored_map = self.env.explored_map.copy()
        sim_env.visit_counts = self.env.visit_counts.copy()
        sim_env.step_count = self.env.step_count
        
        return sim_env


def train_mcts(n_episodes: int = 100, n_simulations: int = 50,
               grid_size: int = 25, num_plants: int = 10, num_obstacles: int = 12,
               save_dir: str = "mcts_models", render: bool = False):
    """
    Train an MCTS agent on the PlantOS environment.
    
    Args:
        n_episodes: Number of episodes to train
        n_simulations: MCTS simulations per action
        grid_size: Environment grid size
        num_plants: Number of plants
        num_obstacles: Number of obstacles
        save_dir: Directory to save statistics
        render: Whether to render during training
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = PlantOSEnv(
        grid_size=grid_size,
        num_plants=num_plants,
        num_obstacles=num_obstacles,
        lidar_range=6,
        lidar_channels=16,
        render_mode='2d' if render else None
    )
    
    # Create MCTS agent
    mcts = MCTS(env, n_simulations=n_simulations, c_param=1.414, max_depth=100)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    exploration_percentages = []
    
    print("=" * 60)
    print(f"Training MCTS Agent")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"MCTS Simulations per action: {n_simulations}")
    print(f"Environment: {grid_size}x{grid_size}, {num_plants} plants, {num_obstacles} obstacles")
    print("=" * 60)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        episode_start_time = time.time()
        
        while not done:
            # Select action using MCTS
            action = mcts.search(obs)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            if render:
                env.render()
                time.sleep(0.01)
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Episode {episode + 1}, Step {step}: "
                      f"Exploration: {info['exploration_percentage']:.1f}%, "
                      f"Reward: {episode_reward:.1f}")
        
        episode_time = time.time() - episode_start_time
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        exploration_percentages.append(info['exploration_percentage'])
        
        # Print episode summary
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes} Complete")
        print(f"{'='*60}")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Exploration: {info['exploration_percentage']:.2f}%")
        print(f"  Thirsty Plants Remaining: {info['thirsty_plants']}")
        print(f"  Time: {episode_time:.2f}s")
        
        # Print running averages (last 10 episodes)
        if episode >= 9:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_exploration = np.mean(exploration_percentages[-10:])
            print(f"\n  Last 10 Episodes Average:")
            print(f"    Reward: {avg_reward:.2f}")
            print(f"    Exploration: {avg_exploration:.2f}%")
        
        print(f"{'='*60}\n")
        
        # Save statistics every 10 episodes
        if (episode + 1) % 10 == 0:
            stats = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'exploration_percentages': exploration_percentages,
                'n_simulations': n_simulations,
                'grid_size': grid_size
            }
            with open(f"{save_dir}/mcts_stats_ep{episode + 1}.pkl", 'wb') as f:
                pickle.dump(stats, f)
            print(f"💾 Statistics saved to {save_dir}/mcts_stats_ep{episode + 1}.pkl\n")
    
    # Save final statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'exploration_percentages': exploration_percentages,
        'n_simulations': n_simulations,
        'grid_size': grid_size
    }
    with open(f"{save_dir}/mcts_stats_final.pkl", 'wb') as f:
        pickle.dump(stats, f)
    
    env.close()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Exploration: {np.mean(exploration_percentages):.2f}% ± {np.std(exploration_percentages):.2f}%")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Statistics saved to: {save_dir}/mcts_stats_final.pkl")
    print("=" * 60)
    
    return stats


def test_mcts(n_episodes: int = 5, n_simulations: int = 100,
              grid_size: int = 25, num_plants: int = 10, num_obstacles: int = 12):
    """
    Test MCTS agent with visualization.
    
    Args:
        n_episodes: Number of test episodes
        n_simulations: MCTS simulations per action (more = better but slower)
        grid_size: Environment grid size
        num_plants: Number of plants
        num_obstacles: Number of obstacles
    """
    # Create environment with rendering
    env = PlantOSEnv(
        grid_size=grid_size,
        num_plants=num_plants,
        num_obstacles=num_obstacles,
        lidar_range=6,
        lidar_channels=16,
        render_mode='2d'
    )
    
    # Create MCTS agent
    mcts = MCTS(env, n_simulations=n_simulations, c_param=1.414, max_depth=100)
    
    print("=" * 60)
    print(f"Testing MCTS Agent")
    print("=" * 60)
    print(f"MCTS Simulations: {n_simulations} (higher = smarter but slower)")
    print("=" * 60)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        while not done:
            # Select action using MCTS
            action = mcts.search(obs)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            # Render
            env.render()
            time.sleep(0.02)  # Slightly slower for visualization
            
            # Print progress
            if step % 20 == 0:
                print(f"  Step {step}: Exploration {info['exploration_percentage']:.1f}%, "
                      f"Reward: {episode_reward:.1f}")
        
        print(f"\n✅ Episode {episode + 1} Complete")
        print(f"   Steps: {step}")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Final Exploration: {info['exploration_percentage']:.2f}%")
        print(f"   Thirsty Plants Remaining: {info['thirsty_plants']}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCTS Trainer for PlantOS')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes')
    parser.add_argument('--simulations', type=int, default=50,
                       help='MCTS simulations per action')
    parser.add_argument('--grid-size', type=int, default=25,
                       help='Grid size')
    parser.add_argument('--plants', type=int, default=10,
                       help='Number of plants')
    parser.add_argument('--obstacles', type=int, default=12,
                       help='Number of obstacles')
    parser.add_argument('--render', action='store_true',
                       help='Render during training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mcts(
            n_episodes=args.episodes,
            n_simulations=args.simulations,
            grid_size=args.grid_size,
            num_plants=args.plants,
            num_obstacles=args.obstacles,
            render=args.render
        )
    else:
        test_mcts(
            n_episodes=args.episodes,
            n_simulations=args.simulations,
            grid_size=args.grid_size,
            num_plants=args.plants,
            num_obstacles=args.obstacles
        )

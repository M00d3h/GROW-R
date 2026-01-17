import numpy as np
from plantos_env import PlantOSEnv
import time
from stable_baselines3 import DQN, A2C
from sb3_contrib import RecurrentPPO
import argparse

def main(model_path: str, model_type: str = 'auto', max_steps_per_episode=1000):
    """
    Run a trained agent in the PlantOS environment with full 2D and 3D visualization.
    
    Args:
        model_path: Path to the trained model zip file
        model_type: Type of model ('dqn', 'ppo', or 'auto' to detect from filename)
        max_steps_per_episode: Maximum steps per episode
    """
    print("Starting PlantOS Environment with 2D and 3D Views")
    print("=" * 60)
    

    if model_type == 'auto':
        if 'dqn' in model_path.lower():
            model_type = 'dqn'
        elif 'ppo' in model_path.lower():
            model_type = 'ppo'
        elif 'a2c' in model_path.lower():
            model_type = 'a2c'
        else:
            print("  Could not auto-detect model type from filename.")
            print("Please specify --model-type dqn, ppo or a2c")
            return
    

    env = PlantOSEnv(grid_size=25, num_plants=10, num_obstacles=20, lidar_range=6, lidar_channels=16, render_mode='human')
    

    if model_type == 'dqn':
        print(" Loading DQN model...")
        model = DQN.load(model_path)
        use_lstm = False
    elif model_type == 'ppo':
        print(" Loading RecurrentPPO model...")
        model = RecurrentPPO.load(model_path)
        use_lstm = True
    elif model_type == 'a2c':
        print(" Loading A2C model...")
        model = A2C.load(model_path)
        use_lstm = False
    else:
        print(f" Unknown model type: {model_type}")
        print("Valid options: 'dqn', 'ppo', 'a2c', or 'auto'")
        return
    
    print(f" Model loaded successfully ({model_type.upper()})")
    
    total_rewards = []
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n📺 Episode {episode}")
            print("-" * 30)
            

            obs, info = env.reset()
            episode_reward = 0
            

            if use_lstm:
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)
            

            for step in range(max_steps_per_episode):

                if use_lstm:
                    action, lstm_states = model.predict(
                        obs, 
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                

                env.render()
                

                if terminated or truncated:
                    break
                

                time.sleep(0.05)
            

            print(f"\nEpisode {episode} finished after {step + 1} steps")
            print(f"Total episode reward: {episode_reward:.2f}")
            print(f"Exploration: {info['exploration_percentage']:.1f}%")
            print(f"Final thirsty plants: {info['thirsty_plants']}")
            
            total_rewards.append(episode_reward)
            
            print("Waiting 2 seconds before next episode...")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n  Environment interrupted by user")
    
    finally:

        env.close()
        

        if total_rewards:
            print("\n" + "=" * 60)
            print(" FINAL SUMMARY")
            print("=" * 60)
            print(f"Episodes completed: {len(total_rewards)}")
            print(f"Average reward: {np.mean(total_rewards):.2f}")
        
        print("Environment closed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent in the PlantOS environment.')
    parser.add_argument('model_path', type=str, help='Path to the trained model zip file')
    parser.add_argument('--model-type', type=str, default='auto', choices=['auto', 'dqn', 'ppo', 'a2c'],
                        help='Type of model: dqn, ppo, a2c, or auto (auto-detect from filename)')
    args = parser.parse_args()
    main(model_path=args.model_path, model_type=args.model_type)
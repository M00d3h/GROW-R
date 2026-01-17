import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback

# Define a callback class for saving models at regular intervals during training
class SaveOnIntervalCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Save the model every 'save_interval' steps
        if self.num_timesteps % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f'model_{self.num_timesteps}')
            self.model.save(save_file)
            if self.verbose > 0:
                print(f'Saving model to {save_file}.zip')
        return True

def visualise_training_logs(metric_name: str, title: str, log_dir: str):
    log_file = os.path.join(log_dir, "progress.csv")
    df = pd.read_csv(log_file)

    window = 50
    rewards = df[metric_name].dropna()
    timesteps = df["time/total_timesteps"].iloc[-len(rewards):]

    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    timesteps = timesteps.iloc[-len(smoothed):]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps/1e6, smoothed, color="deepskyblue", linewidth=2)
    plt.xlabel("Number of Timesteps (millions)")
    plt.ylabel(title)
    plt.title(f"{title} vs Timesteps Smoothed")
    plt.grid(True)
    
    # Save the plot as PNG in the log directory
    output_file = os.path.join(log_dir, f"{title}_smoothed.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")
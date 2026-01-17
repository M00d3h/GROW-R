import gradio as gr
import numpy as np
from plantos_env_new import PlantOSEnvNew
from stable_baselines3 import DQN, A2C
from sb3_contrib import RecurrentPPO
import time
import multiprocessing
import atexit

# --- 3D Viewer Process Management ---
viewer_process = None
update_queue = None

def cleanup_viewer_process():
    """Ensure the 3D viewer process is terminated."""
    global viewer_process, update_queue
    if viewer_process and viewer_process.is_alive():
        print("Terminating 3D viewer process...")
        if update_queue:
            update_queue.put("STOP")
        viewer_process.terminate()
        viewer_process.join(timeout=2)
        print("3D viewer process terminated.")
    viewer_process = None
    update_queue = None

# Register the cleanup function to be called on script exit
atexit.register(cleanup_viewer_process)

def start_3d_viewer(initial_data):
    """Starts the 3D viewer in a separate process."""
    global viewer_process, update_queue
    cleanup_viewer_process()  # Clean up any old process

    print("Starting 3D viewer process...")
    update_queue = multiprocessing.Queue()
    
    from plantos_3d_viewer_new import run_3d_viewer_process
    
    viewer_process = multiprocessing.Process(
        target=run_3d_viewer_process, 
        args=(update_queue, initial_data)
    )
    viewer_process.start()
    print("3D viewer process started.")

def run_simulation_live(model_path, model_type, grid_size, num_plants, num_obstacles, map_generation_algo, max_steps_per_episode=1000):
    """
    Run a trained agent, streaming the 2D view live to Gradio
    and managing a separate 3D viewer window.
    """
    env = PlantOSEnvNew(grid_size=int(grid_size), num_plants=int(num_plants), num_obstacles=int(num_obstacles), 
                        lidar_range=6, lidar_channels=16, render_mode='rgb_array', map_generation_algo=map_generation_algo)
    
    try:
        model_type_lower = model_type.lower()
        if model_type_lower == 'dqn': model = DQN.load(model_path); use_lstm = False
        elif model_type_lower == 'ppo': model = RecurrentPPO.load(model_path); use_lstm = True
        elif model_type_lower == 'a2c': model = A2C.load(model_path); use_lstm = False
        else: raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f" Error loading model: {e}")
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        yield dummy_frame, 0, 0, 0, f"Error: {e}"
        return

    print(f" Model loaded successfully ({model_type.upper()})")
    
    try:
        obs, info = env.reset()
        
        # Start 3D Viewer
        initial_data = {
            'grid_size': env.grid_size,
            'obstacles': list(env.obstacles), # Convert set to list for multiprocessing
            'plants': env.plants,
            'rover_pos': env.rover_pos,
            'stats': info
        }
        start_3d_viewer(initial_data)
        
        episode_reward = 0
        if use_lstm:
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
        
        for step in range(max_steps_per_episode):
            frame_2d = env.render()
            
            if use_lstm:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Send update to 3D viewer
            if update_queue:
                update_data = {
                    'plants': env.plants,
                    'rover_pos': env.rover_pos,
                    'stats': info
                }
                update_queue.put(update_data)
            
            summary = (f"Step: {step + 1}/{max_steps_per_episode} | Reward: {episode_reward:.2f}\n"
                       f"Exploration: {info['exploration_percentage']:.1f}% | Thirsty Plants: {info['thirsty_plants']}\n"
                       f"Rover Position: {info['rover_position']} | Collisions: {info['total_collisions']}\n"
                       f"Explored Cells: {info['explored_cells']}/{info['total_cells']}")
            
            yield frame_2d, episode_reward, info['exploration_percentage'], info['thirsty_plants'], summary
            
            # If the agent is watering, pause the simulation to match the animation duration
            if info.get('is_watering', False):
                time.sleep(1)

            if terminated or truncated:
                print("\nEpisode finished.")
                break
            
            time.sleep(0.05)
    
    finally:
        env.close()
        cleanup_viewer_process()
        print(" Simulation finished and resources cleaned up.")

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# GROW-R - 2D & 3D Live View")
    gr.Markdown("Select a model and path, then click Run. The 2D view will appear below, and a 3D view will open in a new window.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_type = gr.Dropdown(label="Model Type", choices=['DQN', 'PPO', 'A2C'], value='DQN')
            model_path = gr.Textbox(label="Model Path", value="models/dqn_model_weights.zip")
            
            map_generation_algo_dropdown = gr.Dropdown(
                label="Environment Type", 
                choices=[
                    'original', 
                    'maze'
                ], 
                value='original',
                info="original: Open space with scattered obstacles | maze: Wide corridors with irregular walls"
            )
            grid_size_slider = gr.Slider(minimum=10, maximum=50, value=25, step=1, label="Grid Size")
            num_plants_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of Plants")
            num_obstacles_slider = gr.Slider(minimum=0, maximum=200, value=20, step=1, label="Number of Obstacles")

            with gr.Row():
                run_btn = gr.Button("Run Live Simulation", variant="primary")
                stop_btn = gr.Button("Stop Simulation")
        
        with gr.Column(scale=3):
            image_2d_output = gr.Image(label="2D Top-Down View", type="numpy", interactive=False)

    with gr.Row():
        reward_output = gr.Number(label="Total Reward")
        exploration_output = gr.Number(label="Exploration %")
        plants_output = gr.Number(label="Thirsty Plants")
        
    status_output = gr.Textbox(label="Live Episode Stats", lines=6, interactive=False)
    
    run_event = run_btn.click(
        fn=run_simulation_live,
        inputs=[model_path, model_type, grid_size_slider, num_plants_slider, num_obstacles_slider, map_generation_algo_dropdown],
        outputs=[image_2d_output, reward_output, exploration_output, plants_output, status_output]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    demo.launch()
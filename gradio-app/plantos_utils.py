"""
Utility functions for the PlantOS environment, primarily for logging and printing.
"""

def print_step_info(step: int, action: int, reward: float, info: dict):
    """Prints formatted information for a single step."""
    print("-" * 20 + f" Step {step} " + "-" * 20)
    print(f"Action: {action}")
    print(f"Reward: {reward:.2f}")
    print_info_dict(info)
    print("-" * (48 + len(str(step))))


def print_reset_info(info: dict, initial: bool = True):
    """Prints formatted information after an environment reset."""
    title = " Initial State " if initial else " Environment Reset "
    print("=" * 20 + title + "=" * 20)
    print_info_dict(info)
    print("=" * (40 + len(title)))


def print_info_dict(info: dict):
    """Prints the contents of the info dictionary in a readable format."""
    if 'rover_position' in info:
        print(f"Rover Position: {info['rover_position']}")
    if 'thirsty_plants' in info and 'total_plants' in info:
        print(f"Thirsty Spiders: {info['thirsty_plants']} / {info['total_plants']}")
    if 'exploration_percentage' in info and 'explored_cells' in info and 'total_cells' in info:
        # Use .item() to convert numpy types to native Python types for clean printing
        exploration_perc = info['exploration_percentage']
        print(f"Exploration: {exploration_perc:.1f}% ({info['explored_cells']} / {info['total_cells']} cells)")


def print_episode_summary(step: int, info: dict):
    """Prints a summary at the end of an episode."""
    print("\n" + "#" * 20 + " Episode Finished " + "#" * 20)
    print(f"Finished at step {step}.")
    print_info_dict(info)
    print("#" * (60) + "\n")
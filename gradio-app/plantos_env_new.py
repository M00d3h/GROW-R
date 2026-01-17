import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Tuple, Dict, Any, Optional
import os
import math
from plantos_utils import print_reset_info, print_step_info, print_episode_summary
from plantos_3d_viewer_new import PlantOS3DViewer

class PlantOSEnvNew(gym.Env):
    # Observation Channels
    OBSTACLE_CHANNEL = 0
    PLANT_CHANNEL = 1
    THIRST_CHANNEL = 2
    ROVER_CHANNEL = 3
    
    # LIDAR Entity Types
    ENTITY_EMPTY = 0
    ENTITY_OBSTACLE = 1
    ENTITY_PLANT_HYDRATED = 2
    ENTITY_PLANT_THIRSTY = 3
    
    def __init__(self, grid_size: int = 21, num_plants: int = 8, num_obstacles: int = 50, 
                 lidar_range: int = 2, lidar_channels: int = 10, thirsty_plant_prob: float = 0.7,
                 observation_mode: str = "grid", render_mode: Optional[str] = None,
                 map_generation_algo: str = 'original'):
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_plants = num_plants
        self.num_obstacles = num_obstacles
        self.lidar_range = lidar_range
        self.lidar_channels = lidar_channels
        self.thirsty_plant_prob = thirsty_plant_prob
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.map_generation_algo = map_generation_algo
        
        # Action space: 0=North, 1=East, 2=South, 3=West, 4=Water
        self.action_space = spaces.Discrete(5)
        
        # Observation space (LIDAR only, with one-hot encoding)
        # 1 (distance) + 4 (one-hot encoded entity types)
        self.observation_space_per_channel = 1 + 4
        
        # Local visit map parameters
        self.visit_map_size = 5  # 5x5 grid around rover
        self.visit_map_cells = self.visit_map_size * self.visit_map_size  # 25 cells
        
        # Observation space components:
        # - LIDAR: lidar_channels * 5 values
        # - Position: 2 values (x, y)
        # - Local visit map: 25 values (5x5 grid)
        total_obs_size = (self.lidar_channels * self.observation_space_per_channel + 
                         2 +  # position
                         self.visit_map_cells)  # local visit counts
        
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Rewards for A2C - avg_exploration ~ 87%, 10mil steps, with curriculum learning, 512 n_env, 20 minutes
        # self.R_GOAL = 200                   # watering plants
        # self.R_MISTAKE = -20                # watering hydrated plant
        # self.R_INVALID = -11                # invalid move (collision or out of bounds)
        # self.R_WATER_EMPTY = -20            # watering empty space
        # self.R_STEP = -0.1                  # small step penalty to encourage efficiency
        # self.R_EXPLORATION = 10             # Bonus for exploring new cell
        # self.R_REVISIT = -3                 # Small penalty for revisiting explored cell
        # self.R_COMPLETE_EXPLORATION = 100   # Bonus for fully exploring the map

        # Rewards for DQN - avg_exploration ~ 97%, 10mil steps, with curriculum learning, 512 n_env, 9 minutes
        self.R_GOAL = 20
        self.R_MISTAKE = -10
        self.R_INVALID = -5
        self.R_WATER_EMPTY = -5
        self.R_STEP = -0.1
        self.R_EXPLORATION = 10
        self.R_REVISIT = -1
        self.R_COMPLETE_EXPLORATION = 50

        # Rewards for RecurrentPPO - avg_exploration ~ 84%, 3mil steps, with curriculum learning, 128 n_env, 120 minutes, plants are watered but only when encountered, gets stuck towrds the end when near obstacles
        # self.R_GOAL = 50
        # self.R_MISTAKE = -5
        # self.R_INVALID = -2
        # self.R_WATER_EMPTY = -5
        # self.R_STEP = -0.05
        # self.R_EXPLORATION = 5
        # self.R_REVISIT = -0.5
        # self.R_COMPLETE_EXPLORATION = 200
        
        # Internal state variables
        self.rover_pos = None
        self.plants = {}
        self.obstacles = set()
        
        # LIDAR and exploration tracking
        self.explored_map = None
        self.ground_truth_map = None
        
        # ADD: Count-based exploration bonus
        self.visit_counts = None
        
        # Pygame variables for rendering
        self.window = None
        self.clock = None
        self.cell_size = 30
        self.background_img = None
        self.obstacle_img = None
        self.rover_img = None
        self.plant_thirsty_img = None
        self.plant_hydrated_img = None
        self.viewer_3d = None
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        self.collided_with_wall = False
        self.completion_bonus_given = False
        self.total_collisions = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and generate a new random map."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.step_count = 0
        self.collided_with_wall = False
        self.completion_bonus_given = False
        self.total_collisions = 0
        
        # Clear all previous entity locations
        self.plants.clear()
        self.obstacles.clear()
        
        # Generate random map layout
        self._generate_map()
        
        # Initialize exploration map
        self._initialize_exploration()
        
        # Initialize visit counts for count-based exploration
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visit_counts[self.rover_pos[0], self.rover_pos[1]] = 1
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        # If 3D viewer exists, reset its scene to match the new map
        if self.viewer_3d:
            self.viewer_3d.reset_scene()
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Initialize reward for this step
        reward = self.R_STEP  # Base step penalty
        
        # Handle movement actions (0-3)
        if action < 4:
            reward += self._handle_movement(action)
        # Handle watering action (4)
        else:
            reward += self._handle_watering()
        
        # Update LIDAR and exploration
        self._update_lidar()
        
        # Get current observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add watering action flag to info dictionary
        info['is_watering'] = (action == 4)
        
        # Check if episode should terminate
        terminated = self._is_episode_done(info)
        truncated = self.step_count >= self.max_steps
        
        # Check for and award completion bonus
        if info['exploration_percentage'] >= 100 and not self.completion_bonus_given:
            reward += self.R_COMPLETE_EXPLORATION
            self.completion_bonus_given = True
        
        return observation, reward, terminated, truncated, info
    
    def _handle_movement(self, action: int) -> float:
        """
        Handle movement actions with exploration tracking logic.
        
        Args:
            action: Movement action (0=North, 1=East, 2=South, 3=West)
        
        Returns:
            Reward for this movement action
        """
        # Define movement directions: (dx, dy)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # North, East, South, West
        dx, dy = directions[action]
        
        # Calculate new position
        new_x = self.rover_pos[0] + dx
        new_y = self.rover_pos[1] + dy
        
        # Check if new position is valid (within bounds and not an obstacle)
        if (0 <= new_x < self.grid_size and
            0 <= new_y < self.grid_size and
            (new_x, new_y) not in self.obstacles):
            
            was_never_visited = self.visit_counts[new_x, new_y] == 0
            
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 1
            self.rover_pos = (new_x, new_y)
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
            self.visit_counts[new_x, new_y] += 1
            
            if was_never_visited:
                return self.R_EXPLORATION
            else:
                return self.R_REVISIT
        else:
            self.collided_with_wall = True
            self.total_collisions += 1
            return self.R_INVALID
    
    def _handle_watering(self) -> float:
        """Handle the watering action and return the reward."""
        if self.rover_pos in self.plants:
            if self.plants[self.rover_pos]:  # Plant is thirsty
                self.plants[self.rover_pos] = False
                return self.R_GOAL
            else:  # Plant is already hydrated
                return self.R_MISTAKE
        else:
            return self.R_WATER_EMPTY
    
    def _initialize_exploration(self):
        """Initialize the exploration map and ground truth map."""
        self.ground_truth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs_x, obs_y in self.obstacles:
            self.ground_truth_map[obs_x, obs_y] = 1.0
        for (plant_x, plant_y) in self.plants.keys():
            self.ground_truth_map[plant_x, plant_y] = 0.5
        
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
        self._update_lidar()
    
    def _update_lidar(self):
        """Update LIDAR readings based on current rover position."""
        if self.rover_pos is None:
            return
    
    def _is_episode_done(self, info: Dict[str, Any]) -> bool:
        """Check if the episode should terminate."""
        return bool(info['exploration_percentage'] >= 100)
    
    def _get_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array."""
        return self._get_lidar_obs()
    
    def _get_lidar_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array with one-hot encoding for entity types."""
        lidar_size = self.lidar_channels * self.observation_space_per_channel
        position_size = 2
        visit_map_size = self.visit_map_cells
        total_size = lidar_size + position_size + visit_map_size
        
        obs = np.zeros(total_size, dtype=np.float32)
        rover_x, rover_y = self.rover_pos
        
        for i in range(self.lidar_channels):
            angle = (2 * math.pi * i) / self.lidar_channels
            distance = self.lidar_range
            entity_type = self.ENTITY_EMPTY
            
            for r in range(1, self.lidar_range + 1):
                dx = int(r * math.cos(angle))
                dy = int(r * math.sin(angle))
                check_x, check_y = rover_x + dx, rover_y + dy
                
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE
                    break
                
                pos = (check_x, check_y)
                if pos in self.obstacles:
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE
                    break
                elif pos in self.plants:
                    distance = r
                    entity_type = self.ENTITY_PLANT_THIRSTY if self.plants[pos] else self.ENTITY_PLANT_HYDRATED
                    break
            
            start_index = i * self.observation_space_per_channel
            obs[start_index] = distance / self.lidar_range
            one_hot_type = np.zeros(4, dtype=np.float32)
            one_hot_type[entity_type] = 1.0
            obs[start_index + 1 : start_index + 5] = one_hot_type
        
        position_start = lidar_size
        obs[position_start] = rover_x / self.grid_size
        obs[position_start + 1] = rover_y / self.grid_size
        
        visit_map_start = lidar_size + position_size
        visit_map = np.zeros(self.visit_map_cells, dtype=np.float32)
        half_size = self.visit_map_size // 2
        for local_x in range(self.visit_map_size):
            for local_y in range(self.visit_map_size):
                global_x, global_y = rover_x + (local_x - half_size), rover_y + (local_y - half_size)
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    visit_count = min(self.visit_counts[global_x, global_y], 10) / 10.0
                    visit_map[local_x * self.visit_map_size + local_y] = visit_count
                else:
                    visit_map[local_x * self.visit_map_size + local_y] = 1.0
        obs[visit_map_start:visit_map_start + self.visit_map_cells] = visit_map
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the environment."""
        thirsty_count = sum(self.plants.values())
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
        return {
            'rover_position': self.rover_pos,
            'thirsty_plants': thirsty_count,
            'hydrated_plants': len(self.plants) - thirsty_count,
            'total_plants': len(self.plants),
            'step_count': self.step_count,
            'explored_cells': explored_cells,
            'total_cells': total_cells,
            'exploration_percentage': (explored_cells / total_cells) * 100,
            'lidar_range': self.lidar_range,
            'lidar_channels': self.lidar_channels,
            'collided_with_wall': self.collided_with_wall,
            'total_collisions': self.total_collisions
        }
    
    def _generate_map(self):
        """Dispatches the map generation to the selected algorithm."""
        if self.map_generation_algo == 'maze':
            self._generate_map_maze()
        else: # Default to 'original'
            self._generate_map_original()

    def _generate_map_original(self):
        """
        Generate an open environment with randomly scattered obstacles.
        This is much better for exploration learning than narrow mazes.
        """
        # Start with empty grid
        self.obstacles = set()
        
        # Generate random obstacle clusters (more natural than single cells)
        num_obstacle_clusters = self.num_obstacles // 3  # Create clusters instead of individual obstacles
        
        for _ in range(num_obstacle_clusters):
            # Pick random center for cluster
            center_x = random.randint(2, self.grid_size - 3)
            center_y = random.randint(2, self.grid_size - 3)
            
            # Create a small cluster (2x2 or 3x3)
            cluster_size = random.choice([2, 3])
            for dx in range(cluster_size):
                for dy in range(cluster_size):
                    obs_x = center_x + dx - cluster_size // 2
                    obs_y = center_y + dy - cluster_size // 2
                    
                    # Make sure obstacle is within bounds
                    if 0 <= obs_x < self.grid_size and 0 <= obs_y < self.grid_size:
                        self.obstacles.add((obs_x, obs_y))
        
        # Get all available positions (not obstacles)
        available_positions = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ) - self.obstacles
        
        if len(available_positions) < self.num_plants + 1:
            raise ValueError(
                f"Not enough available positions ({len(available_positions)}) to place "
                f"{self.num_plants} plants and 1 rover."
            )
        
        # Place plants randomly
        plant_positions = random.sample(list(available_positions), self.num_plants)
        for plant_pos in plant_positions:
            is_thirsty = random.random() < self.thirsty_plant_prob
            self.plants[plant_pos] = is_thirsty
        available_positions -= set(plant_positions)
        
        # Place rover in a random available position
        self.rover_pos = random.choice(list(available_positions))

    def _generate_map_maze(self):
        """
        Generate a maze with paths that are at least 5 cells wide and irregular diagonal walls.
        """
        # 1. Start with a grid full of obstacles.
        self.obstacles = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))

        # Define a smaller "meta" grid to generate the maze structure
        # Each cell in the meta-grid corresponds to a 5x5 area in the main grid
        meta_w = (self.grid_size - 1) // 6
        meta_h = (self.grid_size - 1) // 6
        
        # Visited cells in the meta-grid
        visited = np.zeros((meta_w, meta_h), dtype=bool)
        
        # Stack for DFS
        stack = []
        
        # Start carving from a random cell in the meta-grid
        start_x, start_y = random.randint(0, meta_w - 1), random.randint(0, meta_h - 1)
        stack.append((start_x, start_y))
        visited[start_x, start_y] = True
        
        # Carve out the initial 5x5 area with irregular shape
        self._carve_irregular_room(start_x, start_y)

        # Randomized DFS on the meta-grid
        while stack:
            cx, cy = stack[-1]
            
            # Get unvisited neighbors (including diagonal)
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Only cardinal directions
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < meta_w and 0 <= ny < meta_h and not visited[nx, ny]:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose a random neighbor
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Carve path to neighbor with irregular shape
                self._carve_irregular_path(cx, cy, nx, ny, dx, dy)
                
                # Carve out the new room at the destination
                self._carve_irregular_room(nx, ny)

                visited[nx, ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Randomly place plants
        available_positions = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size)) - self.obstacles

        # Check if there is enough space for plants and the rover
        if len(available_positions) < self.num_plants + 1:
            print("Warning: Not enough space in the maze for plants and rover. Falling back to original map generation.")
            self._generate_map_original()
            return
        
        plant_positions = random.sample(list(available_positions), self.num_plants)
        for plant_pos in plant_positions:
            # Randomly assign initial plant status
            is_thirsty = random.random() < self.thirsty_plant_prob
            self.plants[plant_pos] = is_thirsty
        available_positions -= set(plant_positions)
        
        # Randomly place rover
        self.rover_pos = random.choice(list(available_positions))
    
    def _carve_irregular_room(self, meta_x, meta_y):
        """Carve out an irregularly shaped room at the given meta coordinates."""
        base_x = meta_x * 6 + 1
        base_y = meta_y * 6 + 1
        
        # Start with a basic 5x5 room
        for i in range(5):
            for j in range(5):
                pos_x = base_x + i
                pos_y = base_y + j
                if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                    self.obstacles.discard((pos_x, pos_y))
        
        # Add irregular extensions (30% chance for each direction)
        if random.random() < 0.3:  # Extend right
            for i in range(2):
                for j in range(2, 4):
                    pos_x = base_x + 5 + i
                    pos_y = base_y + j
                    if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                        self.obstacles.discard((pos_x, pos_y))
        
        if random.random() < 0.3:  # Extend down
            for i in range(2, 4):
                for j in range(2):
                    pos_x = base_x + i
                    pos_y = base_y + 5 + j
                    if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                        self.obstacles.discard((pos_x, pos_y))
        
        # Add diagonal cuts to corners (40% chance)
        if random.random() < 0.4:
            corners = [(0, 0), (4, 0), (0, 4), (4, 4)]
            corner = random.choice(corners)
            pos_x = base_x + corner[0]
            pos_y = base_y + corner[1]
            if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                self.obstacles.add((pos_x, pos_y))
    
    def _carve_irregular_path(self, cx, cy, nx, ny, dx, dy):
        """Carve an irregular path between two meta-grid cells."""
        # For diagonal connections, create a stepped path
        if abs(dx) == 1 and abs(dy) == 1:  # Diagonal connection
            # Create L-shaped or curved path
            mid_x = cx if random.random() < 0.5 else nx
            mid_y = cy if mid_x == cx else ny
            
            # Carve first segment
            self._carve_straight_path(cx, cy, mid_x, mid_y)
            # Carve second segment
            self._carve_straight_path(mid_x, mid_y, nx, ny)
        else:
            # Straight connection but with some irregularity
            self._carve_straight_path(cx, cy, nx, ny)
            
            # Add some random bulges to the path (20% chance)
            if random.random() < 0.2:
                self._add_path_bulge(cx, cy, nx, ny, dx, dy)
    
    def _carve_straight_path(self, cx, cy, nx, ny, width=5):
        """Carve a straight path between two meta-grid cells."""
        if cx == nx:  # Vertical connection
            min_y, max_y = min(cy, ny), max(cy, ny)
            for meta_y in range(min_y, max_y + 1):
                for i in range(width):
                    for j in range(6):
                        pos_x = cx * 6 + 1 + i
                        pos_y = meta_y * 6 + 1 + j
                        if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                            self.obstacles.discard((pos_x, pos_y))
        else:  # Horizontal connection
            min_x, max_x = min(cx, nx), max(cx, nx)
            for meta_x in range(min_x, max_x + 1):
                for i in range(6):
                    for j in range(width):
                        pos_x = meta_x * 6 + 1 + i
                        pos_y = cy * 6 + 1 + j
                        if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                            self.obstacles.discard((pos_x, pos_y))
    
    def _add_path_bulge(self, cx, cy, nx, ny, dx, dy):
        """Add a random bulge to a path for irregularity."""
        mid_x = (cx + nx) // 2
        mid_y = (cy + ny) // 2
        
        # Add a small 2x2 bulge perpendicular to the path direction
        if dx == 0:  # Vertical path, bulge horizontally
            bulge_dir = random.choice([-1, 1])
            for i in range(2):
                for j in range(2):
                    pos_x = mid_x * 6 + 2 + bulge_dir * 2 + i
                    pos_y = mid_y * 6 + 2 + j
                    if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                        self.obstacles.discard((pos_x, pos_y))
        else:  # Horizontal path, bulge vertically
            bulge_dir = random.choice([-1, 1])
            for i in range(2):
                for j in range(2):
                    pos_x = mid_x * 6 + 2 + i
                    pos_y = mid_y * 6 + 2 + bulge_dir * 2 + j
                    if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                        self.obstacles.discard((pos_x, pos_y))
    
    def _add_diagonal_walls(self):
        """Add some diagonal wall patterns throughout the maze."""
        num_diagonals = random.randint(3, 8)
        
        for _ in range(num_diagonals):
            # Pick a random starting point that's currently an obstacle
            obstacle_list = list(self.obstacles)
            if not obstacle_list:
                continue
                
            start_x, start_y = random.choice(obstacle_list)
            
            # Create a short diagonal line (3-6 cells)
            length = random.randint(3, 6)
            direction = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
            for i in range(length):
                wall_x = start_x + i * direction[0]
                wall_y = start_y + i * direction[1]
                
                if (0 <= wall_x < self.grid_size and 0 <= wall_y < self.grid_size):
                    # Only add if it doesn't block essential paths
                    self.obstacles.add((wall_x, wall_y))


    def render(self):
        """Render the environment based on the render_mode."""
        if self.render_mode == 'human':
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
                pygame.display.set_caption("PlantOS Environment")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            frame = self._render_frame()
            self.window.blit(frame, (0, 0))
            pygame.display.flip()
            self.clock.tick(30)
            
            if self.viewer_3d is None:
                self._render_3d() # Initialize 3D viewer
            
            return None

        elif self.render_mode == 'rgb_array':
            frame = self._render_frame()
            return np.transpose(pygame.surfarray.array3d(frame), axes=(1, 0, 2))

    def _render_3d(self):
        """Handles the Ursina 3D rendering."""
        if self.viewer_3d is None:
            self.viewer_3d = PlantOS3DViewer(grid_size=self.grid_size)
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        info = self._get_info()
        stats = {
            'timesteps': info['step_count'],
            'collisions': info['total_collisions'],
            'thirsty_plants': info['thirsty_plants']
        }
        self.viewer_3d.update_scene(self.plants, self.rover_pos, stats)
        self.viewer_3d.render_step()

    def _load_assets(self):
        """Load all Pygame assets."""
        if hasattr(self, '_assets_loaded') and self._assets_loaded:
            return
            
        assets_dir = os.path.dirname(os.path.abspath(__file__))
        
        def get_asset_path(filename):
            # Helper to check for assets in the parent directory as a fallback
            local_path = os.path.join(assets_dir, filename)
            parent_path = os.path.join(assets_dir, '..', 'assets', filename)
            if os.path.exists(local_path):
                return local_path
            elif os.path.exists(parent_path):
                return parent_path
            return None

        asset_paths = {
            'background': get_asset_path('grass_texture.png'),
            'obstacle': get_asset_path('obstacles_texture.png'),
            'rover': get_asset_path('mech_drone_agent.png'),
            'plant_thirsty': get_asset_path('dry_plant_bg.png'),
            'plant_hydrated': get_asset_path('good_plant_bg.png')
        }

        try: self.background_img = pygame.image.load(asset_paths['background']) if asset_paths['background'] else None
        except: self.background_img = None
        if self.background_img: self.background_img = pygame.transform.scale(self.background_img, (self.cell_size, self.cell_size))
        
        try: self.obstacle_img = pygame.image.load(asset_paths['obstacle']) if asset_paths['obstacle'] else None
        except: self.obstacle_img = None
        if self.obstacle_img: self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))

        try: self.rover_img = pygame.image.load(asset_paths['rover']) if asset_paths['rover'] else None
        except: self.rover_img = None
        if self.rover_img: self.rover_img = pygame.transform.scale(self.rover_img, (self.cell_size, self.cell_size))

        try: self.plant_thirsty_img = pygame.image.load(asset_paths['plant_thirsty']) if asset_paths['plant_thirsty'] else None
        except: self.plant_thirsty_img = None
        if self.plant_thirsty_img: self.plant_thirsty_img = pygame.transform.scale(self.plant_thirsty_img, (self.cell_size, self.cell_size))

        try: self.plant_hydrated_img = pygame.image.load(asset_paths['plant_hydrated']) if asset_paths['plant_hydrated'] else None
        except: self.plant_hydrated_img = None
        if self.plant_hydrated_img: self.plant_hydrated_img = pygame.transform.scale(self.plant_hydrated_img, (self.cell_size, self.cell_size))
        
        self._assets_loaded = True

    def _render_frame(self) -> pygame.Surface:
        """Render the current state to a Pygame surface."""
        if not pygame.get_init():
            pygame.init()
        
        self._load_assets()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.background_img: canvas.blit(self.background_img, rect)
                else: pygame.draw.rect(canvas, (34, 139, 34), rect)

        explored_surface = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.SRCALPHA)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.explored_map[x, y] > 0:
                    rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(explored_surface, (200, 200, 200, 100), rect)
        canvas.blit(explored_surface, (0, 0))
        
        for obs_x, obs_y in self.obstacles:
            rect = pygame.Rect(obs_y * self.cell_size, obs_x * self.cell_size, self.cell_size, self.cell_size)
            if self.obstacle_img: canvas.blit(self.obstacle_img, rect)
            else: pygame.draw.rect(canvas, (105, 105, 105), rect)
        
        for (plant_x, plant_y), is_thirsty in self.plants.items():
            rect = pygame.Rect(plant_y * self.cell_size, plant_x * self.cell_size, self.cell_size, self.cell_size)
            img = self.plant_thirsty_img if is_thirsty else self.plant_hydrated_img
            color = (255, 165, 0) if is_thirsty else (0, 255, 0)
            if img: canvas.blit(img, rect)
            else: pygame.draw.rect(canvas, color, rect)
        
        if self.rover_pos:
            rover_x, rover_y = self.rover_pos
            rover_center_x = rover_y * self.cell_size + self.cell_size // 2
            rover_center_y = rover_x * self.cell_size + self.cell_size // 2
            
            for i in range(self.lidar_channels):
                angle = (2 * math.pi * i) / self.lidar_channels
                hit_distance = self.lidar_range
                for r in range(1, self.lidar_range + 1):
                    dx, dy = int(r * math.cos(angle)), int(r * math.sin(angle))
                    check_x, check_y = rover_x + dx, rover_y + dy
                    if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                        hit_distance = r; break
                    if (check_x, check_y) in self.obstacles or (check_x, check_y) in self.plants:
                        hit_distance = r; break
                
                end_x = rover_center_x + int(hit_distance * self.cell_size * math.sin(angle))
                end_y = rover_center_y + int(hit_distance * self.cell_size * math.cos(angle))
                pygame.draw.line(canvas, (100, 100, 255), (rover_center_x, rover_center_y), (end_x, end_y), 1)
                pygame.draw.circle(canvas, (100, 100, 255), (end_x, end_y), 2)
        
        if self.rover_pos:
            rect = pygame.Rect(self.rover_pos[1] * self.cell_size, self.rover_pos[0] * self.cell_size, self.cell_size, self.cell_size)
            if self.rover_img: canvas.blit(self.rover_img, rect)
            else: pygame.draw.rect(canvas, (0, 0, 255), rect)
        
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, x * self.cell_size), (self.grid_size * self.cell_size, x * self.cell_size), 1)
            pygame.draw.line(canvas, (200, 200, 200), (x * self.cell_size, 0), (x * self.cell_size, self.grid_size * self.cell_size), 1)
            
        return canvas
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.viewer_3d:
            self.viewer_3d.close()
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

# Register the environment with Gymnasium
gym.register(
    id='PlantOS-v0',
    entry_point='plantos_env:PlantOSEnvNew',
)
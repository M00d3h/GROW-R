from ursina import *
from ursina import application
import multiprocessing
import time

class PlantOS3DViewer:
    """
    Manages the 3D visualization of the PlantOS environment using Ursina.
    """
    def __init__(self, grid_size: int, cell_size: int = 1):
        """
        Initializes the 3D viewer.
        Note: The Ursina app is a singleton; it can only be initialized once.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.app = Ursina(title='PlantOS 3D View', borderless=False, development_mode=False)
        EditorCamera()
        
        # Scene entities
        self.rover_entity = None
        self.cell_highlighter = None
        self.plant_entities = {}  # Maps (x, y) -> Ursina Entity
        self.obstacle_entities = {} # Maps (x, y) -> Ursina Entity
        self.lidar_lines = []

        # Textures
        self.rover_texture_default = None
        self.rover_texture_water = None

        # HUD for stats
        self.hud_text = Text(
            text="Stats will be shown here",
            position=window.top_right - Vec2(0.05, 0.05),
            origin=(1, 1),
            background=False
        )

        # Setup camera and lighting
        AmbientLight(color=color.rgba(1, 1, 1, 0.8))
        DirectionalLight(color=color.rgba(1, 1, 1, 0.9), direction=(-1, -1, 1))

    def setup_scene(self, obstacles: set, plants: dict, rover_pos: tuple):
        """
        Creates the initial 3D scene with static and dynamic objects.
        """
        grass_texture = 'grass_texture.png'
        self.ground = Entity(
            model='plane',
            scale=(self.grid_size * self.cell_size, 1, self.grid_size * self.cell_size),
            color=color.rgb(34, 139, 34) if not grass_texture else color.white,
            texture=grass_texture,
            texture_scale=(self.grid_size, self.grid_size) if grass_texture else None
        )

        obstacle_texture = 'obstacles_texture.png'
        for obs_pos in obstacles:
            x, y = obs_pos
            self.obstacle_entities[obs_pos] = Entity(
                model='cube',
                color=color.rgb(105, 105, 105) if not obstacle_texture else color.white,
                texture=obstacle_texture,
                position=self._grid_to_world(x, y, 0.5),
                scale=(self.cell_size, self.cell_size, self.cell_size)
            )
        
        self.update_scene(plants, rover_pos)

    def update_scene(self, plants: dict, rover_pos: tuple, stats: dict = None):
        
        if stats:
            self.hud_text.text = (
                f"Timesteps: {stats.get('timesteps', 0)}\n"
                f"Total Collisions: {stats.get('collisions', 0)}\n"
                f"Thirsty Plants: {stats.get('thirsty_plants', 0)}"
            )
        if self.rover_entity is None:
            self.rover_texture_default = 'mech_drone_agent.png'
            self.rover_texture_water = 'mech_drone_water.png'
            self.rover_entity = Entity(
                model='quad', 
                texture=self.rover_texture_default if os.path.exists(self.rover_texture_default) else None,
                color=color.blue if not os.path.exists(self.rover_texture_default) else color.white,
                billboard=True, 
                scale=self.cell_size * 2
            )
        self.rover_entity.position = self._grid_to_world(rover_pos[0], rover_pos[1], 0.5)

        if stats and stats.get('is_watering', False):
            self.trigger_watering_animation()

        if self.cell_highlighter is None:
            self.cell_highlighter = Entity(
                model='cube',
                color=color.green,
                scale=(self.cell_size, 0.1, self.cell_size),
                mode='wireframe'
            )
        self.cell_highlighter.position = self._grid_to_world(rover_pos[0], rover_pos[1], 0.05)

        current_plant_positions = set(self.plant_entities.keys())
        target_plant_positions = set(plants.keys())

        for pos in current_plant_positions - target_plant_positions:
            destroy(self.plant_entities.pop(pos))

        thirsty_texture = 'dry_plant_bg.png'
        hydrated_texture = 'good_plant_bg.png'

        for pos, is_thirsty in plants.items():
            if pos not in self.plant_entities:
                self.plant_entities[pos] = Entity(
                    model='quad',
                    scale=self.cell_size * 2,
                    billboard=True
                )
            
            entity = self.plant_entities[pos]
            entity.position = self._grid_to_world(pos[0], pos[1], 0.5)
            
            if is_thirsty:
                if os.path.exists(thirsty_texture):
                    entity.texture = thirsty_texture
                    entity.color = color.white
                else:
                    entity.texture = None
                    entity.color = color.orange
            else:
                if os.path.exists(hydrated_texture):
                    entity.texture = hydrated_texture
                    entity.color = color.white
                else:
                    entity.texture = None
                    entity.color = color.green

    def trigger_watering_animation(self):
        """Changes texture, rotates for a few seconds, and reverts texture."""
        if not self.rover_entity:
            return

        self.rover_entity.animations.clear()
        self.rover_entity.texture = self.rover_texture_water
        self.rover_entity.animate('rotation_y', self.rover_entity.rotation_y + 360 * 3, duration=1, curve=curve.linear)

        Sequence(
            Wait(1.1), # Wait for slightly longer than the animation duration
            Func(lambda: setattr(self.rover_entity, 'texture', self.rover_texture_default)),
            Func(lambda: setattr(self.rover_entity, 'rotation', Vec3(0, 0, 0))), # Reset rotation to 0 degrees
        ).start()

    def reset_scene(self):
        """Destroys all entities to prepare for a new scene."""
        for entity in self.obstacle_entities.values():
            destroy(entity)
        self.obstacle_entities.clear()

        for entity in self.plant_entities.values():
            destroy(entity)
        self.plant_entities.clear()

        if self.rover_entity:
            destroy(self.rover_entity)
            self.rover_entity = None

        if self.cell_highlighter:
            destroy(self.cell_highlighter)
            self.cell_highlighter = None

    def render_step(self):
        """
        Renders a single frame of the 3D view.
        This should be called in the main application loop.
        """
        self.app.step()

    def _grid_to_world(self, grid_x, grid_y, height):
        """
        Converts grid coordinates to world coordinates for positioning entities.
        Centers the grid around the origin (0,0,0).
        """
        world_x = (grid_x - self.grid_size / 2 + 0.5) * self.cell_size
        world_z = (grid_y - self.grid_size / 2 + 0.5) * self.cell_size
        return (world_x, height * self.cell_size, world_z)

    def close(self):
        """
        Closes the Ursina application window.
        """
        application.quit()

def run_3d_viewer_process(update_queue: multiprocessing.Queue, initial_data: dict):
    """
    Function to be run in a separate process for the 3D viewer.
    """
    viewer = PlantOS3DViewer(grid_size=initial_data['grid_size'])
    viewer.setup_scene(
        initial_data['obstacles'],
        initial_data['plants'],
        initial_data['rover_pos']
    )
    
    # Run the Ursina app loop
    while True:
        if not update_queue.empty():
            message = update_queue.get()
            if message == "STOP":
                break
            else:
                viewer.update_scene(
                    message['plants'],
                    message['rover_pos'],
                    message['stats']
                )
        viewer.render_step()
        time.sleep(0.01) # Small delay to prevent busy-waiting
    
    viewer.close()
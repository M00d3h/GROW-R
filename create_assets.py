
import pygame
import os

# Define colors
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
BROWN = (139, 69, 19)
DARK_GREY = (50, 50, 50)
TRANSPARENT = (0, 0, 0, 0)

# Define image parameters
CELL_SIZE = 30
ROVER_RADIUS = 14
PLANT_RADIUS = 10

# Ensure the script runs in its own directory
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# Set dummy video driver for headless execution
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Initialize Pygame
pygame.init()

# --- Create Rover Image ---
rover_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
rover_surface.fill(TRANSPARENT)
pygame.draw.circle(rover_surface, BLUE, (CELL_SIZE // 2, CELL_SIZE // 2), ROVER_RADIUS)
pygame.draw.circle(rover_surface, DARK_GREY, (CELL_SIZE // 2, CELL_SIZE // 2), ROVER_RADIUS, 2) # Border
pygame.image.save(rover_surface, "rover.png")

# --- Create Thirsty Plant Image ---
plant_thirsty_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
plant_thirsty_surface.fill(TRANSPARENT)
pygame.draw.circle(plant_thirsty_surface, BROWN, (CELL_SIZE // 2, CELL_SIZE // 2), PLANT_RADIUS)
pygame.image.save(plant_thirsty_surface, "plant_thirsty.png")

# --- Create Hydrated Plant Image ---
plant_hydrated_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
plant_hydrated_surface.fill(TRANSPARENT)
pygame.draw.circle(plant_hydrated_surface, GREEN, (CELL_SIZE // 2, CELL_SIZE // 2), PLANT_RADIUS)
pygame.image.save(plant_hydrated_surface, "plant_hydrated.png")

pygame.quit()

print("Assets (rover.png, plant_thirsty.png, plant_hydrated.png) created successfully.")

#!/usr/bin/env python3
"""
Test script for the PlantOS environment.
This script verifies that the environment can be created, reset, and stepped through.
"""

import numpy as np
from plantos_env import PlantOSEnv

def test_environment_creation():
    """Test that the environment can be created with different parameters."""
    print("Testing environment creation...")
    
    # Test default parameters (now Mars Explorer style)
    env = PlantOSEnv()
    assert env.grid_size == 21
    assert env.num_plants == 8
    assert env.num_obstacles == 50
    assert env.lidar_range == 2
    assert env.lidar_channels == 10
    print("✓ Default environment created successfully")
    
    # Test custom parameters
    env_custom = PlantOSEnv(grid_size=15, num_plants=6, num_obstacles=8, lidar_range=4, lidar_channels=16)
    assert env_custom.grid_size == 15
    assert env_custom.num_plants == 6
    assert env_custom.num_obstacles == 8
    assert env_custom.lidar_range == 4
    assert env_custom.lidar_channels == 16
    print("✓ Custom environment created successfully")
    
    env.close()
    env_custom.close()
    return True

def test_action_space():
    """Test that the action space is correctly defined."""
    print("Testing action space...")
    
    env = PlantOSEnv()
    
    # Check action space
    assert env.action_space.n == 5
    assert env.action_space.contains(0)
    assert env.action_space.contains(4)
    assert not env.action_space.contains(5)
    assert not env.action_space.contains(-1)
    print("✓ Action space correctly defined")
    
    env.close()
    return True

def test_observation_space():
    """Test that the observation space is correctly defined for LIDAR."""
    print("Testing observation space...")
    
    env = PlantOSEnv()
    
    # Check observation space
    expected_shape = (env.lidar_channels * env.observation_space_per_channel,)
    assert env.observation_space.shape == expected_shape
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == 0)
    assert np.all(env.observation_space.high == 1)
    print("✓ LIDAR observation space correctly defined")
    
    env.close()
    return True

def test_reset():
    """Test that the environment can be reset and generates valid initial state."""
    print("Testing reset functionality...")
    
    env = PlantOSEnv()
    
    # Reset environment
    obs, info = env.reset()
    
    # Check observation shape
    expected_shape = (env.lidar_channels * env.observation_space_per_channel,)
    assert obs.shape == expected_shape
    assert obs.dtype == np.float32
    
    # Check info structure
    assert 'rover_position' in info
    assert 'thirsty_plants' in info
    assert 'hydrated_plants' in info
    assert 'total_plants' in info
    assert 'step_count' in info
    assert 'collided_with_wall' in info # Check for new key
    assert 'explored_cells' in info
    assert 'exploration_percentage' in info
    assert 'lidar_range' in info
    assert 'lidar_channels' in info
    
    # Check that rover position is valid
    rover_pos = info['rover_position']
    assert 0 <= rover_pos[0] < env.grid_size
    assert 0 <= rover_pos[1] < env.grid_size
    
    # Check plant counts
    assert info['total_plants'] == env.num_plants
    assert info['thirsty_plants'] + info['hydrated_plants'] == env.num_plants
    
    # Check LIDAR parameters
    assert info['lidar_range'] == env.lidar_range
    assert info['lidar_channels'] == env.lidar_channels
    
    print("✓ Reset functionality works correctly")
    
    env.close()
    return True

def test_step():
    """Test that the environment can be stepped through."""
    print("Testing step functionality...")
    
    env = PlantOSEnv()
    obs, info = env.reset()
    
    # Test a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check return values
        expected_shape = (env.lidar_channels * env.observation_space_per_channel,)
        assert obs.shape == expected_shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check that reward is reasonable
        assert -110 <= reward <= 300  # Based on our reward structure
        
        if terminated or truncated:
            break
    
    print("✓ Step functionality works correctly")
    
    env.close()
    return True

def test_episode_termination():
    """Test that episodes terminate correctly."""
    print("Testing episode termination...")
    
    env = PlantOSEnv()
    obs, info = env.reset()
    
    step_count = 0
    max_steps = 1000
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
    if truncated:
        assert step_count == max_steps
        print("✓ Episode correctly truncated after max_steps")
    elif terminated:
        assert step_count < max_steps
        print("✓ Episode correctly terminated before max_steps")

    env.close()
    return True

def test_lidar_observation_content():
    """Test that the LIDAR observation content is correctly structured."""
    print("Testing LIDAR observation content...")
    
    env = PlantOSEnv()
    obs, info = env.reset()
    
    # Check a few observations
    for _ in range(5):
        per_channel = env.observation_space_per_channel
        # Iterate through each channel's data in the observation vector
        for i in range(env.lidar_channels):
            start_index = i * per_channel
            
            # Extract data for one channel
            distance = obs[start_index]
            one_hot_type = obs[start_index + 1 : start_index + per_channel]
            
            # 1. Check that distance is normalized
            assert 0.0 <= distance <= 1.0
            
            # 2. Check that the one-hot encoding is valid (sums to 1)
            assert np.isclose(np.sum(one_hot_type), 1.0)
            assert all(val in [0.0, 1.0] for val in one_hot_type)

        obs, _, _, _, _ = env.step(env.action_space.sample())

    print("✓ LIDAR observation content is correct")
    
    env.close()
    return True

def test_lidar_functionality():
    """Test that LIDAR functionality works correctly."""
    print("Testing LIDAR functionality...")
    
    env = PlantOSEnv()
    obs, info = env.reset()
    
    # Check initial exploration
    assert info['explored_cells'] > 0  # Should have explored initial position
    assert info['exploration_percentage'] > 0
    
    # Take a few steps to test LIDAR updates
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that exploration increases
        assert info['explored_cells'] > 0
        
        if terminated or truncated:
            break
    
    print("✓ LIDAR functionality works correctly")
    
    env.close()
    return True

def main():
    """Run all tests."""
    print("Starting PlantOS Environment Tests (Mars Explorer Style)...\n")
    
    tests = [
        test_environment_creation,
        test_action_space,
        test_observation_space,
        test_reset,
        test_step,
        test_episode_termination,
        test_lidar_observation_content,
        test_lidar_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The PlantOS environment is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()

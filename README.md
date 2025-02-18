# Angry Birds Reinforcement Learning Environment

## Overview

This environment is designed for a Reinforcement Learning (RL) competition where participants train agents to play an Angry Birds-inspired game. 
The environment is implemented using gym and pygame, and features a bird launched from a slingshot to hit a pig positioned at a random location.

## Observation Space

The observation space provides crucial information about the bird's position, velocity, trajectory, and distance from the target pig. 
It is represented as a Box space with 10 continuous values:
spaces.Box(
    low=np.array([0, 0, -30, -30, 0, 0, 0, 0, 0, -np.pi], dtype=np.float32),
    high=np.array([800, 450, 30, 30, 800, 450, 800, 450, 450, np.pi], dtype=np.float32),
    dtype=np.float32
)

import gym
import numpy as np
import random
import pygame  # For GUI

# You'll need to install these:
# pip install gym Box2D-kengz pyglet

# Custom Angry Birds Environment (Simplified Example)
class AngryBirdsEnv(gym.Env):
    def __init__(self, target_x=0.8, target_y=0.5): # Target position
        super(AngryBirdsEnv, self).__init__()

        # Define action space (angle and power of launch)
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)  # Normalized angle and power

        # Define observation space (bird position, target position)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float32) # Normalized bird x, bird y, target x, target y

        self.target_x = target_x
        self.target_y = target_y
        self.bird_x = 0.1 # Initial bird position
        self.bird_y = 0.1
        self.gravity = 0.002
        self.velocity_x = 0
        self.velocity_y = 0
        self.max_steps = 100 # Limit steps per episode
        self.current_step = 0

    def reset(self):
        self.bird_x = 0.1
        self.bird_y = 0.1
        self.velocity_x = 0
        self.velocity_y = 0
        self.current_step = 0
        return np.array([self.bird_x, self.bird_y, self.target_x, self.target_y])

    def step(self, action):
        angle, power = action
        angle = angle * np.pi  # Convert to radians
        power = power * 0.2 # Adjust power scale

        self.velocity_x = power * np.cos(angle)
        self.velocity_y = power * np.sin(angle)

        self.bird_x += self.velocity_x
        self.bird_y += self.velocity_y
        self.velocity_y -= self.gravity

        # Simple collision detection (example)
        if self.bird_y <= 0:
          self.bird_y = 0
          self.velocity_y = 0 # Ground collision
        if self.bird_x > 1:
          self.bird_x = 1
          self.velocity_x = -self.velocity_x * 0.5 # Bounce

        reward = 0
        distance_to_target = np.sqrt((self.bird_x - self.target_x)**2 + (self.bird_y - self.target_y)**2)

        if distance_to_target < 0.05: # Reached target!
            reward = 10
            done = True
        elif self.current_step >= self.max_steps or self.bird_y < 0 and self.velocity_y < 0: # Out of bounds or max steps
            reward = -1
            done = True
        else:
            reward = -distance_to_target * 0.1  # Reward based on distance
            done = False

        self.current_step += 1
        return np.array([self.bird_x, self.bird_y, self.target_x, self.target_y]), reward, done, {}

    def render(self, mode='human'):
        # Basic rendering (you can improve this)
        print(f"Bird: ({self.bird_x:.2f}, {self.bird_y:.2f}), Target: ({self.target_x:.2f}, {self.target_y:.2f})")

# Initialize Pygame
pygame.init()

# Window dimensions
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Angry Birds (Simplified)")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# Game loop
env = AngryBirdsEnv()
observation = env.reset()
running = True
clock = pygame.time.Clock() # For controlling frame rate

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get action (replace with your RL agent's action)
    action = env.action_space.sample()  # Random action for now
    observation, reward, done, info = env.step(action)

    # Clear the screen
    screen.fill(white)

    # Draw target
    target_x = int(observation[2] * width)
    target_y = int(observation[3] * height)
    pygame.draw.circle(screen, red, (target_x, target_y), 10)

    # Draw bird
    bird_x = int(observation[0] * width)
    bird_y = int(observation[1] * height)
    pygame.draw.circle(screen, blue, (bird_x, bird_y), 10)

     # Draw trajectory line (Optional - for visualization)
    if env.velocity_x != 0 and env.velocity_y != 0: # Avoid division by zero
        trajectory_x = bird_x
        trajectory_y = bird_y
        temp_vx = env.velocity_x
        temp_vy = env.velocity_y
        for _ in range(50): # Draw a few points
          trajectory_x += temp_vx * width / 100 # Scale for display
          trajectory_y += temp_vy * height / 100
          temp_vy -= env.gravity
          if trajectory_y > height or trajectory_y < 0 or trajectory_x > width:
            break # Stop drawing if out of bounds
          pygame.draw.circle(screen, black, (int(trajectory_x), int(trajectory_y)), 2)

    # Update the display
    pygame.display.flip()

    if done:
        print("Episode finished")
        observation = env.reset()

    clock.tick(60) # Limit frame rate to 60 FPS (adjust as needed)

env.close()
pygame.quit()


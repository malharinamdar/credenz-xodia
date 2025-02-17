import pygame
import random
import numpy as np
import gym
from gym import spaces

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 450
FPS = 60
SLINGSHOT_POS = (100, 270)

# Load images (update paths as needed)
background = pygame.image.load("UI/images/bg.jpg")
slingshot = pygame.image.load("UI/images/sling.png")
bird_img = pygame.image.load("UI/images/bird.png")
pig_img = pygame.image.load("UI/images/pig.png")

# Scale images
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
slingshot = pygame.transform.scale(slingshot, (60, 120))


class Bird:
    def __init__(self, x, y):
        self.image = pygame.transform.scale(bird_img, (40, 40))
        self.x, self.y = x, y
        self.initial_pos = (x, y)
        self.velocity = [0, 0]
        self.launched = False
        self.trajectory = []
        self.rect = self.image.get_rect()
        self.start_distance = None
        self.max_height = y  # Track maximum height reached
        self.launch_angle = 0  # Track launch angle

    def reset(self):
        self.x, self.y = self.initial_pos
        self.velocity = [0, 0]
        self.launched = False
        self.trajectory = []
        self.start_distance = None
        self.max_height = self.y
        self.launch_angle = 0

    def draw(self, screen):
        self.rect.center = (int(self.x), int(self.y))
        screen.blit(self.image, self.rect)
        if len(self.trajectory) > 1:
            pygame.draw.lines(screen, (255, 0, 0), False, [(int(x), int(y)) for x, y in self.trajectory], 2)

    def launch(self, power_x, power_y):
        if not self.launched:
            # Add randomness to the launch powers
            power_x += random.uniform(-0.5, 0.5)
            power_y += random.uniform(-0.5, 0.5)

            # Scale powers differently and allow for more vertical movement
            self.velocity = [power_x * 4, -power_y * 3.5]
            self.launched = True
            self.trajectory = [(self.x, self.y)]

            # Calculate launch angle
            self.launch_angle = np.arctan2(-self.velocity[1], self.velocity[0])

    def update(self):
        if self.launched:
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            self.velocity[1] += 0.4  # Slightly reduced gravity for higher trajectories

            # Update maximum height
            self.max_height = min(self.max_height, self.y)

            # Add some wind effect
            self.velocity[0] += random.uniform(-0.1, 0.1)

            self.trajectory.append((self.x, self.y))


class Pig:
    def __init__(self, x, y):
        self.image = pygame.transform.scale(pig_img, (40, 40))
        self.x, self.y = x, y
        self.rect = self.image.get_rect()

    def draw(self, screen):
        self.rect.center = (int(self.x), int(self.y))
        screen.blit(self.image, self.rect)

    def reset(self, min_x=500, max_x=700):
        self.x = random.randint(min_x, max_x)
        self.y = random.randint(280, 380)  # Increased vertical range

class AngryBirdsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Angry Birds RL")

        self.bird = Bird(*SLINGSHOT_POS)
        self.pig = Pig(600, 320)

        # Enhanced observation space to include trajectory information
        self.observation_space = None

        # Expanded action space for more varied trajectories
        self.action_space = None

        self.clock = pygame.time.Clock()
        self.current_step = 0
        self.max_steps = 200
        self.min_distance = float('inf')
        self.trajectory_variety = []  # Track trajectory variety

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bird.reset()
        self.pig.reset()
        self.trajectory_variety = []

        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        self.min_distance = np.sqrt(dx * dx + dy * dy)

        return self._get_obs(), {}
    def render(self):
        self.screen.blit(background, (0, 0))
        self.screen.blit(slingshot, SLINGSHOT_POS)
        self.bird.draw(self.screen)
        self.pig.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _get_obs(self): pass

    def step(self, action): pass

if __name__ == "__main__":
    def train_model(): pass
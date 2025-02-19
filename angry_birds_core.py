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
l1_background = pygame.image.load("UI/images/bg.jpg")
l2_background = pygame.image.load("UI/images/bg_2.png")
slingshot = pygame.image.load("UI/images/sling.png")
bird_img = pygame.image.load("UI/images/bird.png")
pig_img = pygame.image.load("UI/images/pig.png")

# Scale images
l1_background = pygame.transform.scale(l1_background, (WIDTH, HEIGHT))
l2_background = pygame.transform.scale(l2_background, (WIDTH, HEIGHT))
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
        self.observation_space = spaces.Box(
        low=np.array([0, 0, -30, -30, 0, 0, 0, -np.pi], dtype=np.float32),
        high=np.array([WIDTH, HEIGHT, 30, 30, WIDTH, HEIGHT, HEIGHT, np.pi], dtype=np.float32),
        dtype=np.float32
        )

        # Expanded action space for more varied trajectories
        self.action_space = spaces.Box(
            low=np.array([1, 1], dtype=np.float32),
            high=np.array([15, 15], dtype=np.float32),  # Increased maximum power
            dtype=np.float32
        )

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

    def _get_obs(self):
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y

        return np.array([
            self.bird.x,
            self.bird.y,
            self.bird.velocity[0],
            self.bird.velocity[1],
            self.pig.x,
            self.pig.y,
            dx,
            dy,
            self.bird.max_height,  # Include maximum height reached
            self.bird.launch_angle  # Include launch angle
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        prev_distance = np.sqrt((self.pig.x - self.bird.x) ** 2 + (self.pig.y - self.bird.y) ** 2)

        if not self.bird.launched:
            power_x, power_y = action
            self.bird.launch(power_x, power_y)

        self.bird.update()

        # Calculate distances and current state
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        current_distance = np.sqrt(dx * dx + dy * dy)
        self.min_distance = min(self.min_distance, current_distance)

        # Enhanced reward system
        reward = 0
        done = False

        # Hit reward
        if current_distance < 40:
            # Bonus for interesting trajectories
            height_bonus = abs(self.bird.max_height - SLINGSHOT_POS[1]) / HEIGHT
            reward = 200.0 + height_bonus * 50
            done = True
        else:
            # Progressive rewards
            distance_improvement = prev_distance - current_distance
            reward += distance_improvement * 0.5

            # Trajectory variety rewards
            height_variety = abs(self.bird.max_height - SLINGSHOT_POS[1]) / HEIGHT
            reward += height_variety * 2

            # Velocity variety reward
            velocity_magnitude = np.sqrt(self.bird.velocity[0] ** 2 + self.bird.velocity[1] ** 2)
            reward += min(velocity_magnitude * 0.1, 5)  # Cap velocity reward

            # Penalize staying too close to the ground or going too high
            if self.bird.y < 50 or self.bird.y > HEIGHT - 30:
                reward -= 1.0

        # Boundary conditions
        if (self.bird.x < 0 or self.bird.x > WIDTH or
                self.bird.y < 0 or self.bird.y > HEIGHT):
            reward = -5.0
            done = True

        # Max steps reached
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, level):
        if level == 1:
            self.screen.blit(l1_background, (0, 0))
        else:

            self.screen.blit(l2_background, (0, 0))
        self.screen.blit(slingshot, SLINGSHOT_POS)
        self.bird.draw(self.screen)
        self.pig.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)

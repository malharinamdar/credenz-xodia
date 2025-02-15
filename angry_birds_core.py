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
background = pygame.image.load("/Users/malhar.inamdar/Downloads/credenz-xodia/UI/images/bg.jpg")
slingshot = pygame.image.load("/Users/malhar.inamdar/Downloads/credenz-xodia/UI/images/sling.png")
bird_img = pygame.image.load("/Users/malhar.inamdar/Downloads/credenz-xodia/UI/images/bird.png")
pig_img = pygame.image.load("/Users/malhar.inamdar/Downloads/credenz-xodia/UI/images/pig.png")

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

    def reset(self):
        self.x, self.y = self.initial_pos
        self.velocity = [0, 0]
        self.launched = False
        self.trajectory = []
        self.start_distance = None

    def draw(self, screen):
        self.rect.center = (int(self.x), int(self.y))
        screen.blit(self.image, self.rect)
        if len(self.trajectory) > 1:
            pygame.draw.lines(screen, (255, 0, 0), False, [(int(x), int(y)) for x, y in self.trajectory], 2)

    def launch(self, power_x, power_y):
        if not self.launched:
            # Scale the powers differently for x and y
            self.velocity = [power_x * 3, -power_y * 2]  # More horizontal movement
            self.launched = True
            self.trajectory = [(self.x, self.y)]

    def update(self):
        if self.launched:
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            self.velocity[1] += 0.5  # Gravity
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
        self.y = random.randint(300, 350)


class AngryBirdsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Angry Birds RL")

        self.bird = Bird(*SLINGSHOT_POS)
        self.pig = Pig(600, 320)  # Initial pig position

        # Adjusted observation space to include distances
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -20, -20, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, 20, 20, WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32),
            dtype=np.float32
        )

        # Adjusted action space for better control
        self.action_space = spaces.Box(
            low=np.array([2, 2], dtype=np.float32),  # Minimum power to ensure movement
            high=np.array([10, 10], dtype=np.float32),
            dtype=np.float32
        )

        self.clock = pygame.time.Clock()
        self.current_step = 0
        self.max_steps = 200
        self.min_distance = float('inf')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bird.reset()
        self.pig.reset(min_x=500, max_x=700)  # Random position within range

        # Calculate initial distance for reward shaping
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        self.min_distance = np.sqrt(dx * dx + dy * dy)

        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        current_distance = np.sqrt(dx * dx + dy * dy)

        return np.array([
            self.bird.x,
            self.bird.y,
            self.bird.velocity[0],
            self.bird.velocity[1],
            self.pig.x,
            self.pig.y,
            dx,  # horizontal distance
            dy   # vertical distance
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        prev_distance = np.sqrt((self.pig.x - self.bird.x) ** 2 + (self.pig.y - self.bird.y) ** 2)

        if not self.bird.launched:
            power_x, power_y = action
            self.bird.launch(power_x, power_y)

        self.bird.update()

        # Calculate new distance and reward
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        current_distance = np.sqrt(dx * dx + dy * dy)

        # Update minimum distance achieved
        self.min_distance = min(self.min_distance, current_distance)

        # Initialize reward
        reward = 0
        done = False

        # Reward shaping
        if current_distance < 40:  # Hit
            reward = 200.0  # Reduced hit reward
            done = True
        else:
            # Progressive reward based on distance improvement
            distance_improvement = prev_distance - current_distance
            reward += distance_improvement * 0.5

            # Penalty for being too high or too low
            if self.bird.y < 100 or self.bird.y > HEIGHT - 50:
                reward -= 0.5

            # Bonus for good trajectory
            if 150 < self.bird.y < 350:
                reward += 0.2

            # Penalty for not moving horizontally enough
            if abs(self.bird.velocity[0]) < 1:
                reward -= 0.3

        # Check boundaries
        if (self.bird.x < 0 or self.bird.x > WIDTH or
                self.bird.y < 0 or self.bird.y > HEIGHT):
            reward = -2.0 * (1 - current_distance / self.min_distance)
            done = True

        # Max steps reached
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        self.screen.blit(background, (0, 0))
        self.screen.blit(slingshot, SLINGSHOT_POS)
        self.bird.draw(self.screen)
        self.pig.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)

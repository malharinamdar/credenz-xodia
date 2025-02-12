import pygame
import random
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 450
FPS = 60
SLINGSHOT_POS = (100, 270)

# Load images
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
            # Normalize the launch power
            power_magnitude = np.sqrt(power_x ** 2 + power_y ** 2)
            normalized_power_x = power_x / (power_magnitude + 1e-8)
            normalized_power_y = power_y / (power_magnitude + 1e-8)

            # Apply launch velocity with better scaling
            self.velocity = [
                normalized_power_x * 15,  # Increased horizontal component
                -normalized_power_y * 10  # Vertical component (negative for upward)
            ]
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
        self.pig = Pig(600, 320)

        # Modified observation space to include normalized distances
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Modified action space for angle and power
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.clock = pygame.time.Clock()
        self.current_step = 0
        self.max_steps = 100
        self.initial_distance = None
        self.best_distance = float('inf')

    def render(self, mode="human"):
        # Draw background
        self.screen.blit(background, (0, 0))

        # Draw slingshot
        self.screen.blit(slingshot, SLINGSHOT_POS)

        # Draw game objects
        self.bird.draw(self.screen)
        self.pig.draw(self.screen)

        # Update display
        pygame.display.flip()

        # Control frame rate
        self.clock.tick(FPS)

        return self.screen

    def _normalize_state(self):
        # Normalize positions and velocities to [-1, 1]
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        return np.array([
            2 * (self.bird.x / WIDTH) - 1,
            2 * (self.bird.y / HEIGHT) - 1,
            self.bird.velocity[0] / 20,  # Normalize velocities
            self.bird.velocity[1] / 20,
            dx / WIDTH,  # Normalize distances
            dy / HEIGHT
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bird.reset()

        # Position pig at one of several predetermined positions
        pig_positions = [
            (500, 320),
            (600, 320),
            (700, 320)
        ]
        pos = random.choice(pig_positions)
        self.pig.x, self.pig.y = pos

        # Calculate initial distance
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        self.initial_distance = np.sqrt(dx ** 2 + dy ** 2)
        self.best_distance = self.initial_distance

        return self._normalize_state(), {}

    def step(self, action):
        self.current_step += 1

        # Get previous state
        prev_dx = self.pig.x - self.bird.x
        prev_dy = self.pig.y - self.bird.y
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2)

        # Launch bird if not launched
        if not self.bird.launched:
            self.bird.launch(action[0] * 10, action[1] * 10)

        # Update bird position
        self.bird.update()

        # Calculate new distance
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        current_distance = np.sqrt(dx ** 2 + dy ** 2)

        # Update best distance
        self.best_distance = min(self.best_distance, current_distance)

        # Initialize reward
        reward = 0
        done = False

        # Hit detection
        if current_distance < 40:
            reward = 100.0 + (self.max_steps - self.current_step)  # Bonus for early hits
            done = True
        else:
            # Distance-based reward
            distance_improvement = (prev_distance - current_distance) * 0.1
            reward += distance_improvement

            # Trajectory shaping rewards
            good_height = 150 < self.bird.y < 350
            good_velocity = abs(self.bird.velocity[0]) > 5  # Encourage horizontal movement

            if good_height:
                reward += 0.2
            if good_velocity:
                reward += 0.3

            # Penalty for being too high or too low
            if self.bird.y < 100 or self.bird.y > HEIGHT - 50:
                reward -= 1.0

        # Out of bounds penalty
        if (self.bird.x < 0 or self.bird.x > WIDTH or
                self.bird.y < 0 or self.bird.y > HEIGHT):
            reward = -10.0 * (current_distance / self.initial_distance)  # Scale penalty
            done = True

        # Max steps reached
        if self.current_step >= self.max_steps:
            reward -= 5.0  # Penalty for timeout
            done = True

        return self._normalize_state(), reward, done, False, {}


def train_model():
    # Create vectorized environment
    env = make_vec_env(AngryBirdsEnv, n_envs=8)  # Increased parallel environments

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.01,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,  # Reduced entropy for more exploitation
        clip_range=0.1,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Larger policy network
                vf=[256, 256]  # Larger value network
            ),
            activation_fn=torch.nn.ReLU
        )
    )

    print("Training started...")
    total_timesteps = 500000  # Increased training time

    # Training with progress updates
    for i in range(5):
        model.learn(total_timesteps=total_timesteps // 5)
        print(f"Training progress: {(i + 1) * 20}%")

    print("Training completed!")
    return model


def run_game(model):
    env = AngryBirdsEnv()
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.bird.launched:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _, _ = env.step(action)
        env.render()  # Now this should work

        if done:
            print(f"Episode finished with reward: {reward}")
            obs, _ = env.reset()
            pygame.time.wait(1000)  # Wait a second before next attempt

    pygame.quit()


if __name__ == "__main__":
    print("Starting training phase...")
    model = train_model()
    print("Starting game phase...")
    run_game(model)
import pygame
import random
import numpy as np
from stable_baselines3 import PPO
from angry_birds_core import AngryBirdsEnv


class TestLevel:
    def __init__(self, level_num):
        self.level_num = level_num
        self.total_attempts = 0
        self.successful_hits = 0
        self.total_reward = 0
        self.scores = []

    def reset_stats(self):
        self.total_attempts = 0
        self.successful_hits = 0
        self.total_reward = 0
        self.scores = []


class TestEnvironment:
    def __init__(self, screen_width=800, screen_height=450):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Angry Birds Model Testing")

        # Initialize levels
        self.levels = {
            1: TestLevel(1),
            2: TestLevel(2),
            3: TestLevel(3)
        }

        self.current_level = 1
        self.episodes_per_level = 10

    def test_model(self, model_path):
        env = AngryBirdsEnv()
        model = PPO.load(model_path)

        for level in range(1, 4):
            self.current_level = level
            self.levels[level].reset_stats()
            print(f"\nTesting Level {level}...")

            for episode in range(self.episodes_per_level):
                reward = self._run_episode(model)
                self.levels[level].scores.append(reward)
                print(f"Episode {episode + 1} Reward: {reward}")

            self._display_level_results(level)

    def _run_episode(self, model):
        # Set up level-specific environment
        if self.current_level == 1:
            return self._run_level_1(model)
        elif self.current_level == 2:
            return self._run_level_2(model)
        else:
            return self._run_level_3(model)

    def _run_level_1(self, model):
        # Basic level: Static target
        pig_pos = (600, 320)
        return self._simulate_episode(model, pig_pos)

    def _run_level_2(self, model):
        # Intermediate: Random target + wind
        x = random.randint(500, 700)
        y = random.randint(280, 380)
        return self._simulate_episode(model, (x, y), wind_strength=0.1)

    def _run_level_3(self, model):
        # Advanced: Multiple targets + strong wind
        targets = [
            (random.randint(500, 700), random.randint(280, 380)),
            (random.randint(500, 700), random.randint(280, 380))
        ]
        return self._simulate_episode(model, targets, wind_strength=0.2)

    def _simulate_episode(self, model, targets, wind_strength=0):
        total_reward = 0
        steps = 0
        max_steps = 200

        # Initial observation
        obs = self._get_initial_observation(targets)

        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)

            # Simulate physics and get new state
            new_obs, reward, done = self._simulate_step(obs, action, targets, wind_strength)
            total_reward += reward

            if done:
                break

            obs = new_obs
            steps += 1

            # Render if needed
            self._render_state(obs, targets)
            pygame.time.wait(50)  # Slow down visualization

        return total_reward

    def _get_initial_observation(self, targets):
        # Convert targets to observation
        if isinstance(targets, tuple):
            targets = [targets]

        obs = np.zeros(10)  # Adjust size based on your observation space
        obs[0] = 100  # Bird initial x
        obs[1] = 270  # Bird initial y
        obs[2] = 0  # Initial velocity x
        obs[3] = 0  # Initial velocity y

        # Add target positions
        for i, (x, y) in enumerate(targets):
            obs[4 + i * 2] = x
            obs[5 + i * 2] = y

        return obs

    def _simulate_step(self, obs, action, targets, wind_strength):
        # Simplified physics simulation
        new_obs = obs.copy()

        # Update bird position and velocity
        new_obs[0] += new_obs[2]  # x position
        new_obs[1] += new_obs[3]  # y position
        new_obs[3] += 0.4  # gravity
        new_obs[2] += random.uniform(-wind_strength, wind_strength)  # wind effect

        # Check if hit any target
        hit = False
        for target_x, target_y in (targets if isinstance(targets, list) else [targets]):
            distance = np.sqrt((new_obs[0] - target_x) ** 2 + (new_obs[1] - target_y) ** 2)
            if distance < 40:  # Hit radius
                hit = True
                break

        # Calculate reward
        reward = 100 if hit else 0

        # Check if done
        done = hit or new_obs[0] < 0 or new_obs[0] > 800 or new_obs[1] < 0 or new_obs[1] > 450

        return new_obs, reward, done

    def _render_state(self, obs, targets):
        self.screen.fill((135, 206, 235))  # Sky blue background

        # Draw bird
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int(obs[0]), int(obs[1])), 10)

        # Draw targets
        for target_x, target_y in (targets if isinstance(targets, list) else [targets]):
            pygame.draw.circle(self.screen, (0, 255, 0),
                               (int(target_x), int(target_y)), 15)

        pygame.display.flip()

    def _display_level_results(self, level):
        level_data = self.levels[level]
        scores = level_data.scores

        print(f"\nLevel {level} Results:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores):.2f}")
        print(f"Min Score: {np.min(scores):.2f}")
        print(f"Success Rate: {(len([s for s in scores if s > 0]) / len(scores)):.2%}")


# Usage example
def test_participant_model(model_path):
    test_env = TestEnvironment()
    test_env.test_model(model_path)
    pygame.quit()


if __name__ == "__main__":
    # Example usage
    model_path = "../angry_birds_model_v2.2"  # Path to participant's trained model
    test_participant_model(model_path)
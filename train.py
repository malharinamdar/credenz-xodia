import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from angry_birds_core import AngryBirdsEnv

def train_model():

    env = make_vec_env(AngryBirdsEnv, n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
    )

    print("Training started...")
    model.learn(total_timesteps=600000)
    print("Training completed!")

    model.save("angry_birds_model_v2.2")
    return model


def check_pig_hit(bird, pig):

    # Update the rectangles to match current positions
    bird.rect.x = bird.x
    bird.rect.y = bird.y
    pig.rect.x = pig.x
    pig.rect.y = pig.y

    # Use pygame's built-in rectangle collision detection
    return bird.rect.colliderect(pig.rect)

def run_game(model):
    env = AngryBirdsEnv()
    obs, _ = env.reset()

    # Level configurations
    LEVEL_1_POSITIONS = [
        (550, 320), (600, 300), (580, 340), (520, 310),
        (590, 330), (540, 300), (570, 320), (510, 340),
        (560, 310), (530, 330)
    ]

    LEVEL_2_POSITIONS = [
        (650, 280), (700, 350), (680, 300), (620, 370),
        (690, 290), (640, 360), (670, 320), (610, 340),
        (660, 380), (630, 310)
    ]

    # Load and scale heart image
    heart_img = pygame.image.load("UI/images/credenz-logo.png")
    heart_img = pygame.transform.scale(heart_img, (30, 30))

    # Initialize game state
    current_level = 1
    current_position = 0
    score = 0
    lives = 5  # Lives per level
    font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current level's positions
        positions = LEVEL_1_POSITIONS if current_level == 1 else LEVEL_2_POSITIONS

        # Set pig position if starting new attempt
        if not env.bird.launched:
            env.pig.x, env.pig.y = positions[current_position]

        # Get AI action
        if not env.bird.launched:
            action, _ = model.predict(obs, deterministic=True)

        # Update game state
        obs, reward, done, _, _ = env.step(action)

        # Render base game
        env.render(level = current_level)

        # Add UI elements on top
        score_text = font.render(f"Score: {score}", True, (205, 41, 0))
        level_text = font.render(f"Level: {current_level}", True, (243, 147, 3))
        position_text = font.render(f"Position: {current_position + 1}/10", True, (255, 255, 255))

        env.screen.blit(score_text, (10, 10))
        env.screen.blit(level_text, (10, 50))
        env.screen.blit(position_text, (10, 90))

        # Draw hearts
        for i in range(lives):
            env.screen.blit(heart_img, (10 + i * 35, 130))

        pygame.display.flip()

        if done:
            pig_hit = check_pig_hit(env.bird, env.pig)

            if pig_hit:
                score += 100
                current_position += 1

                # Check for level completion
                if current_position >= len(positions):
                    if current_level == 1:
                        current_level = 2
                        current_position = 0
                        lives = 3  # Reset lives for new level
                        print("Level 1 completed! Moving to Level 2...")
                    else:
                        print(f"Game completed! Final score: {score}")
                        pygame.time.wait(2000)
                        running = False
                        break
            else:
                current_position += 1
                lives -= 1
                if lives <= 0:

                    print(f"Game Over! Final score: {score}")
                    pygame.time.wait(2000)
                    running = False
                    break

            # Reset environment for next attempt
            # pygame.time.wait(2000)
            obs, _ = env.reset()
            pygame.time.wait(1000)

    pygame.quit()

# def run_game(model):
#     env = AngryBirdsEnv()
#     obs, _ = env.reset()
#
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         if not env.bird.launched:
#             action, _ = model.predict(obs, deterministic=True)
#
#         obs, reward, done, _, _ = env.step(action)
#         env.render()
#
#         if done:
#             print(f"Episode finished with reward: {reward}")
#             obs, _ = env.reset()
#             pygame.time.wait(1000)
#
#     pygame.quit()

if __name__ == "__main__":
    # print("Starting training phase...")
    # model = train_model()

    model = PPO.load("angry_birds_model_v2.2")

    print("Starting game phase...")
    run_game(model)

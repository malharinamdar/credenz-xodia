import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from angry_birds_core import AngryBirdsEnv

def train_model():
    # Create a vectorized environment with multiple instances for training
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
        env.render()

        if done:
            print(f"Episode finished with reward: {reward}")
            obs, _ = env.reset()
            pygame.time.wait(1000)

    pygame.quit()

if __name__ == "__main__":
    print("Starting training phase...")
    model = train_model()
    print("Starting game phase...")
    run_game(model)

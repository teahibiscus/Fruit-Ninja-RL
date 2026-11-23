"""
train_ppo.py - train PPO on FruitNinjaEnv and compare against the greedy baseline.

    pip install gymnasium stable-baselines3
    python train_ppo.py
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from fruit_ninja_env import FruitNinjaEnv, GreedySingleFruitBaseline, evaluate

if __name__ == "__main__":
    venv = make_vec_env(FruitNinjaEnv, n_envs=8, seed=0)
    # Observations are already scaled; only normalize rewards (the ratios stay the same).
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)

    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10,
                gamma=0.99, ent_coef=0.01, learning_rate=3e-4, seed=0, verbose=1)
    model.learn(total_timesteps=2_000_000)
    model.save("ppo_fruit_ninja")

    env = FruitNinjaEnv()
    ppo_policy = lambda obs: model.predict(obs, deterministic=True)[0]
    for name, policy in [("ppo", ppo_policy), ("greedy", GreedySingleFruitBaseline(env.sim))]:
        stats = evaluate(env, policy, n_episodes=50, seed=1000)
        print(f"{name:8s}", {k: round(v, 2) for k, v in stats.items()})

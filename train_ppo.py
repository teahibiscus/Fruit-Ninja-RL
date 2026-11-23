"""
train_ppo.py - train PPO on FruitNinjaEnv, log to TensorBoard, and append one row per run
to results.csv so you can compare runs side by side.

    pip install gymnasium stable-baselines3 tensorboard

Examples:
    python train_ppo.py --name smoke --steps 200000
    python train_ppo.py --name base
    python train_ppo.py --name miss0p1 --miss -0.1
    python train_ppo.py --name bomb20 --bomb -20

Compare runs:
    tensorboard --logdir runs          # then open http://localhost:6006
    python -c "import pandas as pd; print(pd.read_csv('results.csv').to_string())"
"""
import argparse
import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from fruit_ninja_env import FruitNinjaEnv, GreedySingleFruitBaseline, RewardConfig, evaluate

EVAL_EPISODES = 50
EVAL_SEED = 1000          # same eval episodes for every run, so comparisons are paired
RESULTS = "results.csv"


class GameStatsCallback(BaseCallback):
    """Logs game metrics (score, combos, bombs, ...) of finished episodes to TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            s = info.get("episode_stats")
            if s:
                for k in ("score", "fruits", "combos", "bombs", "misses", "steps"):
                    self.logger.record_mean(f"game/{k}", s[k])
        return True


def eval_row(label, env, policy, extra=None):
    stats = evaluate(env, policy, n_episodes=EVAL_EPISODES, seed=EVAL_SEED)
    stats["ep_steps"] = stats.pop("steps")
    return {"name": label, **(extra or {}), **{k: round(v, 2) for k, v in stats.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run", help="label for this run (TensorBoard + results.csv)")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bomb", type=float, default=-10.0)
    ap.add_argument("--miss", type=float, default=-0.5)
    ap.add_argument("--k2", type=float, default=1.0, help="shaping bonus for a 2-fruit swipe")
    ap.add_argument("--combo-scale", type=float, default=1.0, help="bonus = scale * k for k >= 3")
    args = ap.parse_args()

    reward = RewardConfig(bomb=args.bomb, miss=args.miss,
                          combo_bonus_k2=args.k2, combo_bonus_scale=args.combo_scale)
    try:
        import tensorboard  # noqa: F401
        tb_log = "runs"
    except ImportError:
        tb_log = None
        print("tensorboard is not installed, so there will be no learning curves. "
              "Run: pip install tensorboard")

    venv = make_vec_env(FruitNinjaEnv, n_envs=8, seed=args.seed, env_kwargs={"reward": reward})
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)   # obs are already scaled
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99,
                ent_coef=0.01, learning_rate=3e-4, seed=args.seed,
                tensorboard_log=tb_log, verbose=1)
    model.learn(total_timesteps=args.steps, callback=GameStatsCallback(), tb_log_name=args.name)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.name}")

    # ---- evaluate and append to results.csv ----
    env = FruitNinjaEnv(reward=reward)
    env.action_space.seed(0)
    ppo = lambda obs: model.predict(obs, deterministic=True)[0]
    meta = {"seed": args.seed, "train_steps": args.steps, "bomb": args.bomb,
            "miss": args.miss, "k2": args.k2, "combo_scale": args.combo_scale}
    rows = [eval_row(args.name, env, ppo, meta)]

    df = pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame()
    have = set(df["name"]) if "name" in df else set()
    if "greedy" not in have:      # baselines don't depend on reward values: record them once
        rows.append(eval_row("greedy", env, GreedySingleFruitBaseline(env.sim)))
    if "random" not in have:
        rows.append(eval_row("random", env, lambda obs: env.action_space.sample()))
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(RESULTS, index=False)

    cols = ["name", "train_steps", "bomb", "miss", "k2", "score", "combos", "multi",
            "bombs", "misses", "ep_steps"]
    print("\nResults (mean over %d eval episodes; compare on score/combos/bombs, "
          "not return):" % EVAL_EPISODES)
    print(df[cols].sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
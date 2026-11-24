"""record_gif.py - record one episode of a policy playing to play.gif.

    pip install pillow      (already installed with stable-baselines3)
    python record_gif.py                                                  # greedy baseline -> play.gif
    python record_gif.py --model models/smoke                             # trained PPO (loads models/smoke.zip)
    python record_gif.py --model models/smoke --out play1.gif             # trained PPO (loads models/smoke.zip) and outputs as play1.gif
"""
import argparse
from PIL import Image
from fruit_ninja_env import FruitNinjaEnv, GreedySingleFruitBaseline

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None,
                help="path to a saved PPO model, with or without .zip (default: greedy baseline)")
ap.add_argument("--seed", type=int, default=0, help="episode seed passed to env.reset")
ap.add_argument("--out", default="play.gif")

args = ap.parse_args()

env = FruitNinjaEnv(render_mode="rgb_array")
if args.model:
    from stable_baselines3 import PPO   # imported here so the greedy path needs no SB3
    model = PPO.load(args.model)
    policy = lambda obs: model.predict(obs, deterministic=True)[0]
    label = f"PPO {args.model} (deterministic)"
else:
    policy = GreedySingleFruitBaseline(env.sim)
    label = "greedy baseline"

# policy = GreedySingleFruitBaseline(env.sim)   # or: lambda obs: model.predict(obs, deterministic=True)[0]
obs, _ = env.reset(seed=0)
frames = []
for _ in range(400):
    obs, r, term, trunc, info = env.step(policy(obs))
    frames.append(Image.fromarray(env.render()))
    if term or trunc:
        break
frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(len(frames), "frames written to ", args.out)

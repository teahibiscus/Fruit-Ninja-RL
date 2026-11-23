"""record_gif.py - record one episode of a policy playing to play.gif.

    pip install pillow      (already installed with stable-baselines3)
    python record_gif.py
"""
from PIL import Image
from fruit_ninja_env import FruitNinjaEnv, GreedySingleFruitBaseline

env = FruitNinjaEnv(render_mode="rgb_array")
policy = GreedySingleFruitBaseline(env.sim)   # or: lambda obs: model.predict(obs, deterministic=True)[0]
obs, _ = env.reset(seed=0)
frames = []
for _ in range(400):
    obs, r, term, trunc, info = env.step(policy(obs))
    frames.append(Image.fromarray(env.render()))
    if term or trunc:
        break
frames[0].save("play.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)
print(len(frames), "frames written to play.gif")

import re
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import torch

dirpath = os.path.dirname(os.path.abspath(__file__))
method = 'DQN'
filename = 'RL_'+method+'_Results'

file_path = os.path.join(dirpath, filename + '.txt')

with open(file_path, 'r') as file:
    content = file.read()

rewards_pattern = r"Total rewards:\s*\[([^\]]+)\]"
match = re.search(rewards_pattern, content)

if match:
    rewards_raw = match.group(1)
    rewards_cleaned = re.sub(r"tensor\(([^)]+)\)", r"\1", rewards_raw)
    rewards_list = [float(x) for x in rewards_cleaned.split(',')]
    raw_rewards = np.array(rewards_list)

    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return float(x)

    rewards = np.array([to_float(r) for r in raw_rewards])

    episodes = np.arange(len(rewards))

    # avoid log(0)
    eps = 1e-6
    rewards_safe = rewards + eps

    # =========================================================
    #  Main figure
    # =========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        rewards_safe,
        linewidth=0.8,
        alpha=0.85
    )

    plt.yscale("log")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward (log scale)")
    plt.title("Reward Evolution with Rare High-Reward Events")

    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    # =========================================================
    #  Inset: zoom on low-reward region
    # =========================================================
    ax = plt.gca()

    axins = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="upper left",
        borderpad=1.2
    )

    axins.plot(
        episodes,
        rewards,
        linewidth=0.8
    )

    # zoom around the dominant region (~0.3)
    axins.set_ylim(0.25, 3.0)
    axins.set_xlim(0, len(rewards))

    axins.set_title("Zoom: Low-Reward Regime", fontsize=9)
    axins.grid(True, linestyle="--", alpha=0.4)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.tight_layout()
    plt.savefig(os.path.join(dirpath, filename + '_smoothed.pdf'))
    plt.show()
else:
    print("No Total Rewards found in the file.")

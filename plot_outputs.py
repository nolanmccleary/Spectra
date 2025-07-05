import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
ATTACKS = [
    ('ahash_attack', 'AHash', 'red'),
    ('dhash_attack', 'DHash', 'green'),
    ('phash_attack', 'PHash', 'blue'),
    ('pdq_attack',  'PDQ',   'yellow'),
]
JSON_FILE = 'spectra_out.json'
BINS = 40  # you can tweak per‐subplot if you like

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # filter to only those attacks present in the JSON
    present = [ (attack, lab, col) for (attack, lab, col) in ATTACKS if attack in data ]
    betas  = {attack: [] for attack,_,_ in present}
    scales = {attack: [] for attack,_,_ in present}

    for attack,_,_ in present:
        per_image_results = data[attack].get("per_image_results", {})
        for image, result in per_image_results.items():
            pre = result.get("pre_validation", {})
            if pre["success"] == True:
                if "ideal_beta" in pre:
                    betas[attack].append(pre["ideal_beta"])
                if "ideal_scale_factor" in pre:
                    scales[attack].append(pre["ideal_scale_factor"])

    return present, betas, scales


def annotate_stats(ax, data):
    """Compute & annotate μ, median, σ², σ on the given axes."""
    mean   = np.mean(data)
    median = np.median(data)
    var    = np.var(data)
    std    = np.std(data)
    txt = f"μ={mean:.3f}\nmed={median:.3f}\nσ²={var:.3f}\nσ={std:.3f}"
    ax.text(
        0.95, 0.95, txt,
        transform=ax.transAxes,
        fontsize=8,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

def plot_side_by_side(present, betas, scales):
    n = len(present)
    if n == 0:
        print("No known attacks found in JSON.")
        return

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8), squeeze=False)

    # Top row: β distributions
    for col, (key, label, color) in enumerate(present):
        ax = axes[0, col]
        data = betas.get(key, [])
        if data:
            bins = np.linspace(min(data), max(data), BINS)
            ax.hist(data, bins=bins, color=color, edgecolor='black')
            annotate_stats(ax, data)
        ax.set_title(f'{label} β')
        ax.set_xlabel('β')
        ax.set_ylabel('Count')
        if data:
            ax.set_xlim(min(data), max(data))

    # Bottom row: scale‐factor distributions
    for col, (key, label, color) in enumerate(present):
        ax = axes[1, col]
        data = scales.get(key, [])
        if data:
            bins = np.linspace(min(data), max(data), BINS)
            ax.hist(data, bins=bins, color=color, edgecolor='black')
            annotate_stats(ax, data)
        ax.set_title(f'{label} Scale Factor')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Count')
        if data:
            ax.set_xlim(min(data), max(data))

    plt.tight_layout()
    plt.show()

def main():
    present, betas, scales = load_data(JSON_FILE)
    plot_side_by_side(present, betas, scales)

if __name__ == '__main__':
    main()

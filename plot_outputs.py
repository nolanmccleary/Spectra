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
BINS = 20  # tweak per‐subplot if you like

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    attacks = data.keys()

    betas = {}
    scales = {}

    print(attacks)

    for attack in attacks:
        betas[attack] = []
        scales[attack] = []
        
        per_image_results = data[attack]["per_image_results"]
        images = per_image_results.keys()
        
        for image in images:
            betas[attack].append(per_image_results[image]["pre_validation"]["ideal_beta"])
            scales[attack].append(per_image_results[image]["pre_validation"]["ideal_scale_factor"])

    return betas, scales


def annotate_stats(ax, data):
    """Compute & annotate μ, median, σ², σ on the given axes."""
    mean = np.mean(data)
    median = np.median(data)
    var = np.var(data)
    std = np.std(data)
    text = (
        f"μ={mean:.3f}\n"
        f"med={median:.3f}\n"
        f"σ²={var:.3f}\n"
        f"σ={std:.3f}"
    )
    ax.text(
        0.95, 0.95, text,
        transform=ax.transAxes,
        fontsize=8,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

def plot_side_by_side(betas, scales):
    fig, axes = plt.subplots(2, len(ATTACKS), figsize=(16, 8))
    
    # Top row: β distributions
    for col, (key, label, color) in enumerate(ATTACKS):
        ax = axes[0, col]
        data = betas[key]
        if data:
            bins = np.linspace(min(data), max(data), BINS)
            ax.hist(data, bins=bins, color=color, edgecolor='black')
            annotate_stats(ax, data)
        ax.set_title(f'{label} β')
        ax.set_xlabel('β')
        ax.set_ylabel('Count')
        ax.set_xlim(min(data, default=0), max(data, default=1))
    
    # Bottom row: scale‐factor distributions
    for col, (key, label, color) in enumerate(ATTACKS):
        ax = axes[1, col]
        data = scales[key]
        if data:
            bins = np.linspace(min(data), max(data), BINS)
            ax.hist(data, bins=bins, color=color, edgecolor='black')
            annotate_stats(ax, data)
        ax.set_title(f'{label} Scale Factor')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Count')
        ax.set_xlim(min(data, default=0), max(data, default=1))
    
    plt.tight_layout()
    plt.show()

def main():
    betas, scales = load_data(JSON_FILE)
    plot_side_by_side(betas, scales)

if __name__ == '__main__':
    main()

"""Plotting functions for the report."""

import numpy as np
import matplotlib.pyplot as plt

from metrics import f1_score


# --- Helper function for Plot 1 ---
def plot_distribution(ax, p):
    """Plots the probability distribution histogram on a given axes."""
    ax.hist(p, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of predicted probabilities (p)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5)


# --- Helper function for Plot 2 ---
def plot_f1_curve(ax, thresholds, scores):
    """Plots the F1 vs. threshold curve on a given axes."""
    ax.plot(thresholds, scores)
    ax.set_title("F1 vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")


def tune_threshold_plot(p, y_true, threshold_strategy, make_plots, save_plots=False):
    """Use instead of tune_threshold if you require saving plots."""
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true contains values other than 0 and 1.")

    if threshold_strategy == "unique":
        thresholds = np.unique(p)
    elif threshold_strategy == "rounded":
        thresholds = np.unique(np.round(p, 4))
    elif threshold_strategy == "linspace":
        thresholds = np.linspace(0, 1, 1000)
    elif threshold_strategy == "quantile":
        thresholds = np.quantile(p, np.linspace(0, 1, 1000))
    else:
        raise ValueError("Invalid threshold strategy")

    scores = [f1_score(y_true, (p >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(scores)
    best_t = thresholds[best_idx]
    best_f1 = scores[best_idx]

    if make_plots:
        # --- 1. Create the side-by-side plot for display ---
        fig_display, axes_display = plt.subplots(1, 2, figsize=(14, 5))

        # Use helper functions to plot on the display axes
        plot_distribution(axes_display[0], p)
        plot_f1_curve(axes_display[1], thresholds, scores)

        plt.tight_layout()
        plt.show()  # Show the combined plot

        # --- 2. Create separate plots for saving ---
        if save_plots:
            # Create and save the first plot
            fig_hist, ax_hist = plt.subplots(figsize=(7, 5))  # New, single figure
            plot_distribution(ax_hist, p)  # Use helper
            plt.tight_layout()
            fig_hist.savefig("probability_distribution.png")
            plt.close(fig_hist)  # Close figure to free memory

            # Create and save the second plot
            fig_f1, ax_f1 = plt.subplots(figsize=(7, 5))  # Another new, single figure
            plot_f1_curve(ax_f1, thresholds, scores)  # Use helper
            plt.tight_layout()
            fig_f1.savefig("f1_vs_threshold.png")
            plt.close(fig_f1)  # Close figure to free memory

    print(len(thresholds), "unique thresholds tested")
    print("Best threshold: ", best_t)
    print("Best F1 score: ", best_f1)

    return best_t, best_f1

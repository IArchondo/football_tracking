import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd


def create_base_pitch(pitch_length=106, pitch_width=68):
    """Create a dataframe with all possible combinations of coordinates in pitch

    Args:
        pitch_length (float): pitch length in meters
        half_pitch_width (float): pitch width in meters

    Returns:
        pd.DataFrame: dataframe with all combinations
    """
    list_x = list(range(-pitch_length // 2, (pitch_length // 2) + 1))

    list_y = list(range(-pitch_width // 2, (pitch_width // 2) + 1))

    comb = list(itertools.product(list_x, list_y))

    base_pitch = pd.DataFrame(comb, columns=["x", "y"])

    return base_pitch


def plot_pitch(pitch_width=68, pitch_length=106):
    """Plot pitch

    Args:
        pitch_width (int, optional): pitch width in meters. Defaults to 68.
        pitch_length (int, optional): pitch length in meters. Defaults to 106.

    Returns:
        fig,ax : plot
    """
    fig, ax = plt.subplots(figsize=(8, 60 / pitch_length * 8))
    ax.set_facecolor("mediumseagreen")
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color="red")
    points_whole_ax = 8 * 72 * 0.8
    radius = 9.15
    points_radius = 2 * radius / 1.0 * points_whole_ax
    ax.scatter(
        pitch_length / 2,
        pitch_width / 2,
        marker="o",
        s=points_radius,
        facecolors="none",
        edgecolors="red",
    )
    ax.plot(
        [16.5, 16.5],
        [(pitch_width - 40.3) / 2, pitch_width - (pitch_width - 40.3) / 2],
        color="red",
    )
    ax.plot(
        [pitch_length - 16.5, pitch_length - 16.5],
        [(pitch_width - 40.3) / 2, pitch_width - (pitch_width - 40.3) / 2],
        color="red",
    )
    ax.plot(
        [pitch_length - 16.5, pitch_length],
        [(pitch_width - 40.3) / 2, (pitch_width - 40.3) / 2],
        color="red",
    )
    ax.plot(
        [pitch_length - 16.5, pitch_length],
        [
            pitch_width - (pitch_width - 40.3) / 2,
            pitch_width - (pitch_width - 40.3) / 2,
        ],
        color="red",
    )
    ax.plot(
        [0, 16.5], [(pitch_width - 40.3) / 2, (pitch_width - 40.3) / 2], color="red"
    )
    ax.plot(
        [0, 16.5],
        [
            pitch_width - (pitch_width - 40.3) / 2,
            pitch_width - (pitch_width - 40.3) / 2,
        ],
        color="red",
    )

    ax.plot([5.5, 5.5], [22, pitch_width - (22)], color="red")
    ax.plot([0, 5.5], [22, 22], color="red")
    ax.plot([0, 5.5], [pitch_width - 22, pitch_width - 22], color="red")

    ax.scatter(0, pitch_width - 27.5, color="blue")
    ax.scatter(0, 27.5, color="blue")

    ax.plot(
        [pitch_length - 5.5, pitch_length - 5.5], [22, pitch_width - (22)], color="red"
    )
    ax.plot([pitch_length - 5.5, pitch_length], [22, 22], color="red")
    ax.plot(
        [pitch_length - 5.5, pitch_length],
        [pitch_width - 22, pitch_width - 22],
        color="red",
    )

    ax.scatter(pitch_length, pitch_width - 27.5, color="blue")
    ax.scatter(pitch_length, 27.5, color="blue")

    return fig, ax

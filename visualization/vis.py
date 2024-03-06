import matplotlib.pyplot as plt
import os

def visualize_incorrect_pose(vis_dir, pose, gt_class, pred_class):
    """
    documentation later
    """
    plt.clf()
    x_values = pose[::2]
    y_values = pose[1::2]

    pairs = [
        (0, 1),
        (1, 2),
        (0, 2),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 5),
        (3, 6),
        (5, 7),
        (3, 8),
        (5, 8),
        (6, 9),
        (8, 10),
        (8, 11),
        (7, 12),
        (10, 13),
        (11, 14),
        (13, 15),
        (14, 16)
    ]

    for idx1, idx2 in pairs:
        plt.plot((x_values[idx1], x_values[idx2]), (y_values[idx1], y_values[idx2]))
    plt.gca().set_title(f"gt: {gt_class}, pred: {pred_class}")
    plt.savefig(vis_dir)

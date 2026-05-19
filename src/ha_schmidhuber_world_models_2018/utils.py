from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def save_input_reconstruction_grid(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Save paired ``(input, reconstruction)`` images into a single grid PNG.

    Each cell shows the input on top and its reconstruction directly below, so
    pairs can be compared at a glance.
    """
    if inputs.shape != reconstructions.shape:
        raise ValueError(
            f"inputs {inputs.shape} and reconstructions {reconstructions.shape} "
            "must have the same shape."
        )
    if inputs.ndim != 4:
        raise ValueError(
            "inputs must have shape (batch, height, width, channels); "
            f"got {inputs.shape}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = inputs.shape[0]
    num_cols = int(np.ceil(np.sqrt(batch_size)))
    num_pair_rows = int(np.ceil(batch_size / num_cols))
    fig, axes = plt.subplots(
        2 * num_pair_rows,
        num_cols,
        figsize=(num_cols * 1.5, num_pair_rows * 3.0),
        squeeze=False,
    )
    for i in range(num_pair_rows * num_cols):
        pair_row, col = divmod(i, num_cols)
        ax_input = axes[2 * pair_row, col]
        ax_recon = axes[2 * pair_row + 1, col]
        ax_input.axis("off")
        ax_recon.axis("off")
        if i < batch_size:
            ax_input.imshow(np.clip(inputs[i], 0.0, 1.0))
            ax_recon.imshow(np.clip(reconstructions[i], 0.0, 1.0))
            if pair_row == 0:
                ax_input.set_title("input", fontsize=8)
            if pair_row == num_pair_rows - 1 or i + num_cols >= batch_size:
                ax_recon.set_xlabel("reconstruction", fontsize=8)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_transition_prediction_grid(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    next_inputs: np.ndarray,
    predictions: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Save one transition per column with clearly labeled image rows."""
    arrays = {
        "inputs": inputs,
        "reconstructions": reconstructions,
        "next_inputs": next_inputs,
        "predictions": predictions,
    }
    first_shape = inputs.shape
    for name, array in arrays.items():
        if array.shape != first_shape:
            raise ValueError(
                f"{name} shape {array.shape} must match inputs shape {first_shape}."
            )
        if array.ndim != 4:
            raise ValueError(
                f"{name} must have shape (batch, height, width, channels); "
                f"got {array.shape}."
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = inputs.shape[0]
    row_labels = (
        "Input frame",
        "VAE reconstruction",
        "True next frame",
        "Predicted next frame",
    )
    image_rows = (inputs, reconstructions, next_inputs, predictions)
    fig, axes = plt.subplots(
        len(row_labels),
        batch_size,
        figsize=(batch_size * 1.8, len(row_labels) * 1.8),
        squeeze=False,
    )
    for row_idx, (label, images) in enumerate(zip(row_labels, image_rows, strict=True)):
        for col_idx in range(batch_size):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(images[col_idx], 0.0, 1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(f"Transition {col_idx + 1}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=52, va="center")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.96))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

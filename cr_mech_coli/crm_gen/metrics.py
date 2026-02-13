#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics for comparing synthetic images with real microscope images.

This module provides three key metrics:

- Color/Intensity Distribution (histogram comparison)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

These metrics are used to evaluate and optimize synthetic image generation
to match real microscope images.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tifffile
from skimage import img_as_float
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import matplotlib.pyplot as plt
import json


def load_image(path: Path) -> np.ndarray:
    """
    Load a TIFF image and convert to float [0,1].

    Args:
        path (Path): Path to the image file.

    Returns:
        np.ndarray: Image as float array with values in [0,1].
    """
    img = tifffile.imread(path)
    return img_as_float(img)


def compute_color_distribution(
    image1: np.ndarray, image2: np.ndarray, bins: int = 256
) -> Dict[str, np.ndarray]:
    """
    Compute and compare color/intensity distributions of two images.

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        bins (int): Number of histogram bins.

    Returns:
        dict: Dictionary containing 'hist1', 'hist2', 'bin_edges',
            'histogram_diff', and 'histogram_distance' (L1 norm).
    """
    # Flatten images to 1D
    flat1 = image1.flatten()
    flat2 = image2.flatten()

    # Compute histograms with same bins
    hist1, bin_edges = np.histogram(flat1, bins=bins, range=(0, 1), density=True)
    hist2, _ = np.histogram(flat2, bins=bins, range=(0, 1), density=True)

    # Compute difference and distance
    hist_diff = np.abs(hist1 - hist2)
    hist_distance = np.sum(hist_diff)

    return {
        "hist1": hist1,
        "hist2": hist2,
        "bin_edges": bin_edges,
        "histogram_diff": hist_diff,
        "histogram_distance": float(hist_distance),
    }


def compute_ssim(
    image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures the structural similarity between images, considering
    luminance, contrast, and structure. Values range from -1 to 1, where
    1 indicates perfect similarity.

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        data_range (float): Data range of the images (1.0 for float images).

    Returns:
        dict: Dictionary containing 'ssim' score (higher is better, max=1.0).
    """
    # Handle grayscale vs RGB
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        # RGB image - compute SSIM per channel and average
        ssim_score = ssim(image1, image2, data_range=data_range, channel_axis=2)
    else:
        # Grayscale image
        ssim_score = ssim(image1, image2, data_range=data_range)

    return {"ssim": float(ssim_score)}


def compute_psnr(
    image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR measures the ratio between the maximum possible signal power
    and the power of corrupting noise. Higher values indicate better quality.
    Typical range: 20-50 dB (higher is better).

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        data_range (float): Data range of the images (1.0 for float images).

    Returns:
        dict: Dictionary containing 'psnr' value in dB (higher is better).
    """
    psnr_value = psnr(image1, image2, data_range=data_range)

    return {"psnr": float(psnr_value)}


def compute_all_metrics(
    original: np.ndarray, synthetic: np.ndarray, bins: int = 256
) -> Dict[str, any]:
    """
    Compute all metrics comparing original and synthetic images.

    This is the main method that computes all three metrics in one call.

    Args:
        original (np.ndarray): Original microscope image (float [0,1]).
        synthetic (np.ndarray): Synthetic image (float [0,1]).
        bins (int): Number of bins for histogram.

    Returns:
        dict: Dictionary containing 'color_distribution', 'ssim', 'psnr',
            and 'summary' statistics.
    """
    # Handle shape mismatches (grayscale vs RGB)
    # Convert RGB to grayscale if needed
    if len(original.shape) == 3 and original.shape[2] == 3:
        # RGB original - convert to grayscale
        original = np.mean(original, axis=2)

    if len(synthetic.shape) == 3 and synthetic.shape[2] == 3:
        # RGB synthetic - convert to grayscale
        synthetic = np.mean(synthetic, axis=2)

    # Ensure images have same shape after conversion
    if original.shape != synthetic.shape:
        raise ValueError(
            f"Image shapes don't match after conversion: {original.shape} vs {synthetic.shape}"
        )

    # Compute individual metrics
    color_dist = compute_color_distribution(original, synthetic, bins=bins)
    ssim_result = compute_ssim(original, synthetic)
    psnr_result = compute_psnr(original, synthetic)

    # Combine all results
    results = {
        "color_distribution": color_dist,
        "ssim": ssim_result,
        "psnr": psnr_result,
        "summary": {
            "histogram_distance": color_dist["histogram_distance"],
            "ssim_score": ssim_result["ssim"],
            "psnr_db": psnr_result["psnr"],
        },
    }

    return results


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): Metrics dictionary from compute_all_metrics().
        output_path (Path): Path where to save the JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {
        "summary": metrics["summary"],
        "color_distribution": {
            "hist1": metrics["color_distribution"]["hist1"].tolist(),
            "hist2": metrics["color_distribution"]["hist2"].tolist(),
            "bin_edges": metrics["color_distribution"]["bin_edges"].tolist(),
            "histogram_diff": metrics["color_distribution"]["histogram_diff"].tolist(),
            "histogram_distance": metrics["color_distribution"]["histogram_distance"],
        },
        "ssim": metrics["ssim"],
        "psnr": metrics["psnr"],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def plot_metrics(
    original: np.ndarray,
    synthetic: np.ndarray,
    metrics: Dict,
    output_path: Optional[Path] = None,
    title: str = "Image Comparison Metrics",
) -> None:
    """
    Create a visualization of all metrics.

    Args:
        original (np.ndarray): Original image.
        synthetic (np.ndarray): Synthetic image.
        metrics (dict): Metrics dictionary from compute_all_metrics().
        output_path (Path): If provided, save plot to this path. If None, plot is
            only displayed.
        title (str): Title for the plot.
    """
    # Handle shape mismatches (grayscale vs RGB) for visualization
    # Convert RGB to grayscale if needed
    if len(original.shape) == 3 and original.shape[2] == 3:
        original = np.mean(original, axis=2)

    if len(synthetic.shape) == 3 and synthetic.shape[2] == 3:
        synthetic = np.mean(synthetic, axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Color/Intensity Distribution (top-left)
    ax = axes[0, 0]
    bin_centers = (
        metrics["color_distribution"]["bin_edges"][:-1]
        + metrics["color_distribution"]["bin_edges"][1:]
    ) / 2
    ax.plot(
        bin_centers,
        metrics["color_distribution"]["hist1"],
        label="Original",
        color="#2E86AB",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        bin_centers,
        metrics["color_distribution"]["hist2"],
        label="Synthetic",
        color="#A23B72",
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("Intensity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Color Distribution\nL1 Distance: {metrics['summary']['histogram_distance']:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. SSIM Score (top-right)
    ax = axes[0, 1]
    ssim_score = metrics["summary"]["ssim_score"]
    color = (
        "#2ECC71" if ssim_score > 0.9 else "#F39C12" if ssim_score > 0.7 else "#E74C3C"
    )
    ax.text(
        0.5,
        0.5,
        f"{ssim_score:.4f}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.set_title("SSIM Score\n(Structural Similarity)", fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.25,
        "Range: -1 to 1 (higher is better)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
        color="gray",
    )
    ax.axis("off")

    # 3. PSNR Value (bottom-left)
    ax = axes[1, 0]
    psnr_value = metrics["summary"]["psnr_db"]
    color = (
        "#2ECC71" if psnr_value > 30 else "#F39C12" if psnr_value > 20 else "#E74C3C"
    )
    ax.text(
        0.5,
        0.5,
        f"{psnr_value:.2f} dB",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.set_title("PSNR\n(Peak Signal-to-Noise Ratio)", fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.25,
        "Typical range: 20-50 dB (higher is better)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
        color="gray",
    )
    ax.axis("off")

    # 4. Side-by-side image comparison (bottom-right)
    ax = axes[1, 1]
    # Both images are now grayscale after conversion at function start
    combined = np.hstack([original, synthetic])
    ax.imshow(combined, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Visual Comparison", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Add labels
    h, w = original.shape[:2]
    ax.text(w // 2, h + 10, "Original", ha="center", fontsize=9, fontweight="bold")
    ax.text(w + w // 2, h + 10, "Synthetic", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.close()

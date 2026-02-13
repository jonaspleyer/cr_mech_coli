#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic plotting for synthetic image optimization.

Generates side-by-side comparisons of real and synthetic microscope images
with intensity histograms and region-specific quality metrics (SSIM, PSNR,
histogram distance).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Dict
import random
import shutil
import tifffile as tiff
from tqdm import tqdm

from .scene import create_synthetic_scene
from .metrics import load_image, compute_all_metrics


def _create_histogram_subplot(
    ax: plt.Axes,
    real_img: np.ndarray,
    synth_img: np.ndarray,
    title: str,
    histogram_distance: float,
    ssim_score: float,
    psnr_db: float,
    exclude_zeros: bool = False,
) -> None:
    """
    Create a histogram subplot comparing intensity distributions with metrics overlay.

    Args:
        ax (plt.Axes): Matplotlib axes to plot on.
        real_img (np.ndarray): Real microscope image (float [0,1]).
        synth_img (np.ndarray): Synthetic image (float [0,1]).
        title (str): Title for the subplot.
        histogram_distance (float): L1 histogram distance between images.
        ssim_score (float): SSIM score between images.
        psnr_db (float): PSNR value in dB.
        exclude_zeros (bool): If True, exclude zero-valued pixels from histograms.
    """
    real_values = real_img.flatten()
    synth_values = synth_img.flatten()

    if exclude_zeros:
        real_values = real_values[real_values > 0]
        synth_values = synth_values[synth_values > 0]

    hist_real, bins_real = np.histogram(
        real_values, bins=256, range=(0, 1), density=True
    )
    hist_synth, _ = np.histogram(synth_values, bins=256, range=(0, 1), density=True)
    bin_centers = (bins_real[:-1] + bins_real[1:]) / 2

    ax.plot(
        bin_centers, hist_real, label="Real", color="#2E86AB", linewidth=2, alpha=0.8
    )
    ax.plot(
        bin_centers,
        hist_synth,
        label="Synthetic",
        color="#A23B72",
        linewidth=2,
        alpha=0.8,
    )

    ax.set_xlabel("Intensity", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlim(0, 1)

    metrics_text = f"Hist: {histogram_distance:.2f}\nSSIM: {ssim_score:.3f}\nPSNR: {psnr_db:.1f} dB"
    ax.text(
        0.02,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def extract_masked_region(
    image: np.ndarray, mask: np.ndarray, region: str
) -> np.ndarray:
    """
    Extract foreground or background region from image using mask.

    Args:
        image (np.ndarray): Input image (float [0,1]).
        mask (np.ndarray): Segmentation mask (2D or 3D).
        region (str): Region to extract: 'foreground' or 'background'.

    Returns:
        np.ndarray: Image with the non-selected region zeroed out.
    """
    if len(mask.shape) == 3:
        mask = np.max(mask, axis=2)

    extracted = image.copy()

    if region == "foreground":
        extracted[mask == 0] = 0.0
    elif region == "background":
        extracted[mask > 0] = 0.0
    else:
        raise ValueError(f"Invalid region: {region}")

    return extracted


def generate_detailed_plots(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    per_image_metrics: List[Dict],
    output_dir: Path,
    n_vertices: int,
) -> None:
    """
    Generate detailed per-image plots with region-specific analysis.

    Creates one plot per image pair showing full, background, and foreground
    histogram comparisons along with image and mask visualizations.

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        params (Dict): Optimized synthetic parameters (bg_base_brightness, etc.).
        per_image_metrics (List[Dict]): Per-image metrics from compute_final_metrics().
        output_dir (Path): Output directory for saving plots.
        n_vertices (int): Number of vertices for cell shape extraction.
    """
    print(f"\nGenerating detailed plots for {len(image_pairs)} images...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    temp_dir = output_dir / "temp_detailed_plots"
    temp_dir.mkdir(exist_ok=True)

    try:
        metrics_lookup = {m["image_name"]: m for m in per_image_metrics}

        for real_img_path, mask_path in tqdm(
            image_pairs, desc="Creating detailed plots"
        ):
            synthetic_img, synthetic_mask = create_synthetic_scene(
                microscope_image_path=real_img_path,
                segmentation_mask_path=mask_path,
                output_dir=temp_dir,
                n_vertices=n_vertices,
                bg_base_brightness=params["bg_base_brightness"],
                bg_gradient_strength=params["bg_gradient_strength"],
                bac_halo_intensity=params["bac_halo_intensity"],
                bg_noise_scale=int(params["bg_noise_scale"]),
                psf_sigma=params["psf_sigma"],
                peak_signal=params["peak_signal"],
                gaussian_sigma=params["gaussian_sigma"],
            )

            if synthetic_img.dtype == np.uint8:
                synthetic_img = synthetic_img.astype(np.float64) / 255.0

            real_img = load_image(real_img_path)
            original_mask = tiff.imread(mask_path)

            real_fg = extract_masked_region(real_img, original_mask, "foreground")
            real_bg = extract_masked_region(real_img, original_mask, "background")
            synth_fg = extract_masked_region(
                synthetic_img, synthetic_mask, "foreground"
            )
            synth_bg = extract_masked_region(
                synthetic_img, synthetic_mask, "background"
            )

            img_metrics = metrics_lookup[real_img_path.name]

            fig = plt.figure(figsize=(14, 12))
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

            ax_full = fig.add_subplot(gs[0, 0])
            _create_histogram_subplot(
                ax_full,
                real_img,
                synthetic_img,
                "Full Image",
                img_metrics["full_histogram_distance"],
                img_metrics["full_ssim_score"],
                img_metrics["full_psnr_db"],
            )

            ax_bg = fig.add_subplot(gs[0, 1])
            _create_histogram_subplot(
                ax_bg,
                real_bg,
                synth_bg,
                "Background Region",
                img_metrics["bg_histogram_distance"],
                img_metrics["bg_ssim_score"],
                img_metrics["bg_psnr_db"],
                exclude_zeros=True,
            )

            ax_fg = fig.add_subplot(gs[1, 0])
            _create_histogram_subplot(
                ax_fg,
                real_fg,
                synth_fg,
                "Foreground Region (Bacteria)",
                img_metrics["fg_histogram_distance"],
                img_metrics["fg_ssim_score"],
                img_metrics["fg_psnr_db"],
                exclude_zeros=True,
            )

            ax_images = fig.add_subplot(gs[1, 1])
            real_display = (
                real_img if len(real_img.shape) == 2 else np.mean(real_img, axis=2)
            )
            synth_display = (
                synthetic_img
                if len(synthetic_img.shape) == 2
                else np.mean(synthetic_img, axis=2)
            )
            combined = np.hstack([real_display, synth_display])
            ax_images.imshow(combined, cmap="gray", vmin=0, vmax=1)
            ax_images.set_title("Image Comparison", fontsize=10, fontweight="bold")
            ax_images.axis("off")
            h, w = real_display.shape
            ax_images.text(
                w // 2, h + 15, "Real", ha="center", fontsize=9, fontweight="bold"
            )
            ax_images.text(
                w + w // 2,
                h + 15,
                "Synthetic",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

            ax_real_mask = fig.add_subplot(gs[2, 0])
            mask_display = (
                original_mask
                if len(original_mask.shape) == 2
                else np.max(original_mask, axis=2)
            )
            ax_real_mask.imshow(mask_display, cmap="tab20", interpolation="nearest")
            ax_real_mask.set_title(
                "Real Segmentation Mask", fontsize=10, fontweight="bold"
            )
            ax_real_mask.axis("off")

            ax_synth_mask = fig.add_subplot(gs[2, 1])
            synth_mask_display = (
                synthetic_mask
                if len(synthetic_mask.shape) == 2
                else np.max(synthetic_mask, axis=2)
            )
            ax_synth_mask.imshow(
                synth_mask_display, cmap="tab20", interpolation="nearest"
            )
            ax_synth_mask.set_title(
                "Synthetic Segmentation Mask", fontsize=10, fontweight="bold"
            )
            ax_synth_mask.axis("off")

            fig.suptitle(
                f"{real_img_path.stem}\n"
                f"Params: bg={params['bg_base_brightness']:.2f}, "
                f"grad={params['bg_gradient_strength']:.4f}, "
                f"halo={params['bac_halo_intensity']:.2f}, "
                f"noise={int(params['bg_noise_scale'])}",
                fontsize=12,
                fontweight="bold",
            )

            plot_path = plots_dir / f"plot_{real_img_path.stem}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"  Saved detailed plots to: {plots_dir}")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def generate_comparison_plot(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    output_dir: Path,
    n_vertices: int,
    num_examples: int = 3,
):
    """
    Generate comparison plot showing original vs synthetic for example images.

    Creates a single figure with histogram comparisons and side-by-side image
    views for a sample of image pairs.

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        params (Dict): Optimized synthetic parameters (bg_base_brightness, etc.).
        output_dir (Path): Output directory for saving the plot.
        n_vertices (int): Number of vertices for cell shape extraction.
        num_examples (int): Number of example images to show.
    """
    print("\nGenerating comparison plot...")

    if len(image_pairs) <= 2:
        example_pairs = image_pairs
        num_examples = len(image_pairs)
    else:
        random.seed(42)
        num_examples = 3
        example_pairs = random.sample(image_pairs, num_examples)

    print(f"  Showing {num_examples} example(s)...")

    temp_dir = output_dir / "temp_plot"
    temp_dir.mkdir(exist_ok=True)

    try:
        examples = []
        for real_img_path, mask_path in tqdm(example_pairs, desc="Generating examples"):
            synthetic_img, _ = create_synthetic_scene(
                microscope_image_path=real_img_path,
                segmentation_mask_path=mask_path,
                output_dir=temp_dir,
                n_vertices=n_vertices,
                bg_base_brightness=params["bg_base_brightness"],
                bg_gradient_strength=params["bg_gradient_strength"],
                bac_halo_intensity=params["bac_halo_intensity"],
                bg_noise_scale=int(params["bg_noise_scale"]),
                psf_sigma=params["psf_sigma"],
                peak_signal=params["peak_signal"],
                gaussian_sigma=params["gaussian_sigma"],
            )

            if synthetic_img.dtype == np.uint8:
                synthetic_img = synthetic_img.astype(np.float64) / 255.0

            real_img = load_image(real_img_path)
            metrics = compute_all_metrics(real_img, synthetic_img)

            examples.append(
                {
                    "name": real_img_path.stem,
                    "real": real_img,
                    "synthetic": synthetic_img,
                    "metrics": metrics["summary"],
                }
            )

        nrows = 2
        ncols = num_examples
        fig = plt.figure(figsize=(6 * num_examples, 8))

        gs = GridSpec(
            nrows, ncols, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.2
        )

        for idx, example in enumerate(examples):
            real_display = example["real"]
            synth_display = example["synthetic"]

            if len(real_display.shape) == 3:
                real_display = np.mean(real_display, axis=2)
            if len(synth_display.shape) == 3:
                synth_display = np.mean(synth_display, axis=2)

            ax_hist = fig.add_subplot(gs[0, idx])

            hist_real, bins_real = np.histogram(
                real_display.flatten(), bins=100, range=(0, 1), density=True
            )
            hist_synth, _ = np.histogram(
                synth_display.flatten(), bins=100, range=(0, 1), density=True
            )
            bin_centers = (bins_real[:-1] + bins_real[1:]) / 2

            ax_hist.plot(
                bin_centers,
                hist_real,
                label="Original",
                color="#2E86AB",
                linewidth=2,
                alpha=0.8,
            )
            ax_hist.plot(
                bin_centers,
                hist_synth,
                label="Synthetic",
                color="#A23B72",
                linewidth=2,
                alpha=0.8,
            )

            ax_hist.set_xlabel("Intensity", fontsize=9)
            ax_hist.set_ylabel("Density", fontsize=9)
            ax_hist.legend(loc="upper right", fontsize=9)
            ax_hist.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            ax_hist.set_xlim(0, 1)

            ax_img = fig.add_subplot(gs[1, idx])

            combined = np.hstack([real_display, synth_display])

            ax_img.imshow(combined, cmap="gray", vmin=0, vmax=1)
            ax_img.axis("off")

            title = f"{example['name']}\n"
            title += f"Hist: {example['metrics']['histogram_distance']:.2f} | "
            title += f"SSIM: {example['metrics']['ssim_score']:.3f} | "
            title += f"PSNR: {example['metrics']['psnr_db']:.1f} dB"
            ax_img.set_title(title, fontsize=11, fontweight="bold", pad=10)

            h, w = real_display.shape
            ax_img.text(
                w // 2, h + 20, "Original", ha="center", fontsize=10, fontweight="bold"
            )
            ax_img.text(
                w + w // 2,
                h + 20,
                "Synthetic",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        title_str = "Optimized Parameters: "
        title_str += f"bg={params['bg_base_brightness']:.2f}, "
        title_str += f"grad={params['bg_gradient_strength']:.4f}, "
        title_str += f"halo={params['bac_halo_intensity']:.2f}, "
        title_str += f"noise={int(params['bg_noise_scale'])}"

        fig.suptitle(title_str, fontsize=14, fontweight="bold", y=0.995)

        plt.tight_layout()

        plot_path = output_dir / "comparison_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {plot_path.name}")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

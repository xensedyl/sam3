"""
SAM3 Image Segmentation & Visualization Demo

Usage:
    # Basic: text prompt segmentation
    python examples/test.py

    # Custom image + prompt
    python examples/test.py --image assets/images/truck.jpg --prompt "truck"

    # Multiple text prompts
    python examples/test.py --image assets/images/groceries.jpg --prompt "apple" "banana" "bottle"

    # Adjust confidence threshold
    python examples/test.py --prompt "gamepad" --threshold 0.3

    # Save to specific path (default: examples/output/)
    python examples/test.py --prompt "gamepad" --output results/

    # Use CPU (slow, for debugging only)
    python examples/test.py --prompt "gamepad" --device cpu
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from PIL import Image

# ─── SAM3 imports ───────────────────────────────────────────────────────────────
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


# ─── Visualization helpers (self-contained, no heavy deps) ─────────────────────
def generate_distinct_colors(n: int) -> list:
    """Generate n visually distinct colors via HSV spacing."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        color = matplotlib.colors.hsv_to_rgb([hue, 0.9, 0.9])
        colors.append(color)
    return colors


def overlay_mask(ax, mask: np.ndarray, color, alpha: float = 0.45):
    """Overlay a binary mask with the given color on an axes."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    rgb = to_rgb(color)
    overlay[..., :3] = rgb
    overlay[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(overlay)


def draw_bbox(ax, box_xyxy, color, label_text: str = ""):
    """Draw a bounding box (x1,y1,x2,y2 absolute) on an axes."""
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    rect = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=2, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label_text:
        ax.text(
            x1, max(y1 - 6, 0), label_text,
            color="white", fontsize=9, weight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )


def visualize_and_save(
    image: Image.Image,
    results: dict,
    prompt_text: str,
    save_path: str,
    show: bool = False,
):
    """
    Visualize segmentation results on a single image/prompt and save to file.

    Args:
        image:       PIL image
        results:     dict with keys 'masks', 'boxes', 'scores' (tensors)
        prompt_text: the text prompt used
        save_path:   output file path
        show:        whether to call plt.show()
    """
    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]
    n_objects = len(scores)

    colors = generate_distinct_colors(max(n_objects, 1))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ── Left panel: original image ─────────────────────────────────────────
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=16, weight="bold")
    axes[0].axis("off")

    # ── Right panel: segmentation overlay ──────────────────────────────────
    axes[1].imshow(image)
    if n_objects == 0:
        axes[1].text(
            0.5, 0.5, "No objects detected",
            transform=axes[1].transAxes, fontsize=20,
            ha="center", va="center", color="red", weight="bold",
        )
    else:
        for i in range(n_objects):
            color = colors[i]
            mask_np = masks[i].squeeze(0).cpu().numpy()
            box_np = boxes[i].cpu().numpy()
            score = scores[i].item()

            overlay_mask(axes[1], mask_np, color)
            draw_bbox(axes[1], box_np, color, label_text=f"#{i} {score:.2f}")

    axes[1].set_title(
        f'Prompt: "{prompt_text}"  |  Found {n_objects} object(s)',
        fontsize=16, weight="bold",
    )
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"  -> saved to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def visualize_multi_prompt(
    image: Image.Image,
    all_results: list[tuple[str, dict]],
    save_path: str,
    show: bool = False,
):
    """
    Visualize results from multiple prompts in a single overview figure.

    Args:
        image:       PIL image
        all_results: list of (prompt_text, result_dict)
        save_path:   output file path
    """
    n_prompts = len(all_results)
    ncols = min(n_prompts + 1, 4)
    nrows = 1 + (n_prompts) // 4  # +1 for original
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :] if axes.ndim == 1 else axes
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    axes_flat = axes.flatten()

    # First cell: original
    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original Image", fontsize=14, weight="bold")
    axes_flat[0].axis("off")

    # Remaining cells: one per prompt
    for idx, (prompt_text, results) in enumerate(all_results, start=1):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        ax.imshow(image)

        masks = results["masks"]
        boxes = results["boxes"]
        scores = results["scores"]
        n_obj = len(scores)
        colors = generate_distinct_colors(max(n_obj, 1))

        for i in range(n_obj):
            mask_np = masks[i].squeeze(0).cpu().numpy()
            box_np = boxes[i].cpu().numpy()
            score = scores[i].item()
            overlay_mask(ax, mask_np, colors[i])
            draw_bbox(ax, box_np, colors[i], label_text=f"#{i} {score:.2f}")

        ax.set_title(f'"{prompt_text}" ({n_obj} obj)', fontsize=13, weight="bold")
        ax.axis("off")

    # Hide unused axes
    for j in range(len(all_results) + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"  -> multi-prompt overview saved to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def save_individual_masks(
    image: Image.Image,
    results: dict,
    prompt_text: str,
    output_dir: str,
):
    """Save each detected mask as a separate binary PNG."""
    masks = results["masks"]
    scores = results["scores"]
    n_objects = len(scores)
    if n_objects == 0:
        return

    mask_dir = os.path.join(output_dir, "masks", prompt_text.replace(" ", "_"))
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(n_objects):
        mask_np = masks[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np, mode="L")
        mask_path = os.path.join(mask_dir, f"mask_{i:03d}_score{scores[i].item():.2f}.png")
        mask_img.save(mask_path)

    print(f"  -> {n_objects} individual mask(s) saved to {mask_dir}/")


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 Image Segmentation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", type=str, default=os.path.join(sam3_root, "examples", "20260306-171430.jpg"),
        help="Path to input image (default: examples/gamepad.jpg)",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="+", default=["robot"],
        help='Text prompt(s) for segmentation, e.g. --prompt "cat" "dog"',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(sam3_root, "examples", "output"),
        help="Output directory for results (default: examples/output/)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Also save individual binary masks as PNG files",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display matplotlib window (interactive mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ── Device & precision setup ───────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        # Enable TF32 for Ampere GPUs (A100, RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Use bfloat16 for faster inference
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    torch.inference_mode().__enter__()

    # ── Load model ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("SAM3 Image Segmentation Demo")
    print("=" * 60)

    print(f"\n[1/3] Loading SAM3 model (device={device}) ...")
    t0 = time.time()
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)
    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # ── Load image ─────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    print(f"       Image size: {w} x {h}")

    t0 = time.time()
    inference_state = processor.set_image(image)
    print(f"       Image encoded in {time.time() - t0:.2f}s")

    # ── Run segmentation for each prompt ───────────────────────────────────
    print(f"\n[3/3] Running segmentation (threshold={args.threshold}) ...")
    all_results = []

    for prompt in args.prompt:
        print(f'\n  Prompt: "{prompt}"')
        processor.reset_all_prompts(inference_state)

        t0 = time.time()
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        elapsed = time.time() - t0

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        n_objects = len(scores)

        print(f"    Found {n_objects} object(s) in {elapsed:.2f}s")
        for i in range(n_objects):
            score = scores[i].item()
            box = boxes[i].cpu().tolist()
            print(f"    - obj #{i}: score={score:.3f}, bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

        all_results.append((prompt, output))

        # Save per-prompt visualization
        safe_name = prompt.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(args.output, f"seg_{safe_name}.png")
        visualize_and_save(image, output, prompt, save_path, show=args.show)

        # Optionally save individual masks
        if args.save_masks:
            save_individual_masks(image, output, prompt, args.output)

    # ── Multi-prompt overview ──────────────────────────────────────────────
    if len(args.prompt) > 1:
        overview_path = os.path.join(args.output, "overview_all_prompts.png")
        visualize_multi_prompt(image, all_results, overview_path, show=args.show)

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"All results saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

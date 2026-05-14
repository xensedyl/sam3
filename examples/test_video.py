"""
SAM3 Video Segmentation & Tracking Demo

Usage:
    # Basic: text prompt tracking on default video
    python examples/test_video.py

    # Custom video + prompt
    python examples/test_video.py --video examples/飞书20260331-110729.mp4 --prompt "Bottled water on the conveyor belt"

    # Multiple text prompts
    python examples/test_video.py --video assets/videos/bedroom.mp4 --prompt "bed" "pillow" "lamp"

    # Use JPEG frame folder as input
    python examples/test_video.py --video assets/videos/0001 --prompt "person"

    # Adjust confidence threshold
    python examples/test_video.py --prompt "person" --threshold 0.5

    # Limit the number of frames to process
    python examples/test_video.py --prompt "person" --max-frames 100

    # Save output as MP4 video (requires ffmpeg)
    python examples/test_video.py --prompt "person" --save-video

    # Save individual frame PNGs
    python examples/test_video.py --prompt "person" --save-frames
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb

# ─── SAM3 imports ───────────────────────────────────────────────────────────────
import sam3
from sam3.model_builder import build_sam3_video_model

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


# ─── Visualization helpers ──────────────────────────────────────────────────────
def generate_distinct_colors(n: int) -> list:
    """Generate n visually distinct colors via HSV spacing."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        color = matplotlib.colors.hsv_to_rgb([hue, 0.9, 0.9])
        colors.append(color)
    return colors


def overlay_masks_on_frame(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    obj_ids: np.ndarray,
    scores: np.ndarray,
    colors: list,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay binary masks and bounding boxes on a BGR frame.

    Args:
        frame_bgr:  (H, W, 3) uint8 BGR frame
        masks:      (N, H, W) bool masks
        obj_ids:    (N,) object IDs
        scores:     (N,) confidence scores
        colors:     list of RGB tuples for each object
        alpha:      mask transparency

    Returns:
        (H, W, 3) uint8 BGR frame with overlays
    """
    vis = frame_bgr.copy()
    n_objects = len(obj_ids)

    for i in range(n_objects):
        mask = masks[i]
        obj_id = int(obj_ids[i])
        score = float(scores[i])
        color_idx = obj_id % len(colors)
        rgb = np.array(colors[color_idx]) * 255

        # Overlay mask
        colored_mask = np.zeros_like(vis, dtype=np.float32)
        colored_mask[..., 0] = rgb[2]  # BGR order
        colored_mask[..., 1] = rgb[1]
        colored_mask[..., 2] = rgb[0]
        mask_3d = mask[:, :, np.newaxis].astype(np.float32)
        vis = (vis.astype(np.float32) * (1 - mask_3d * alpha) + colored_mask * mask_3d * alpha)
        vis = vis.astype(np.uint8)

        # Draw bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_color, 2)
            label = f"#{obj_id} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), bgr_color, -1)
            cv2.putText(
                vis, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return vis


def save_overview_figure(
    original_frame: np.ndarray,
    overlay_frame: np.ndarray,
    frame_idx: int,
    prompt_text: str,
    n_objects: int,
    save_path: str,
):
    """Save a side-by-side comparison figure for a single frame."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Frame #{frame_idx} (Original)", fontsize=16, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f'Prompt: "{prompt_text}"  |  Tracking {n_objects} object(s)',
        fontsize=16, weight="bold",
    )
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ─── Video I/O helpers ──────────────────────────────────────────────────────────
def read_video_frames(video_path: str, max_frames: int = -1) -> tuple:
    """
    Read raw video frames for visualization.

    Returns:
        frames: list of (H, W, 3) uint8 BGR frames
        fps: float
    """
    if os.path.isdir(video_path):
        # JPEG frame folder
        from sam3.model.io_utils import IMAGE_EXTS
        frame_names = sorted(
            [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in IMAGE_EXTS],
            key=lambda p: int(os.path.splitext(p)[0]) if os.path.splitext(p)[0].isdigit() else p,
        )
        if 0 < max_frames < len(frame_names):
            frame_names = frame_names[:max_frames]
        frames = []
        for fn in frame_names:
            img = cv2.imread(os.path.join(video_path, fn))
            if img is not None:
                frames.append(img)
        fps = 30.0  # default for image folders
        return frames, fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        if 0 < max_frames <= len(frames):
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(frames: list, output_path: str, fps: float):
    """Write a list of BGR frames to an MP4 file."""
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 Video Segmentation & Tracking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", type=str, default=os.path.join(sam3_root, "assets", "videos", "bedroom.mp4"),
        help="Path to input video (MP4) or JPEG frame folder (default: assets/videos/bedroom.mp4)",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="+", default=["bed"],
        help='Text prompt(s) for segmentation, e.g. --prompt "person" "car"',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=-1,
        help="Maximum number of frames to process (-1 = all frames)",
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
        "--save-video", action="store_true",
        help="Save output as MP4 video with overlaid segmentation",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save individual frame PNGs with overlaid segmentation",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display results with OpenCV window (press Q to quit)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ── Device & precision setup ───────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    torch.inference_mode().__enter__()

    # ── Load model ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("SAM3 Video Segmentation & Tracking Demo")
    print("=" * 60)

    print(f"\n[1/4] Loading SAM3 video model (device={device}) ...")
    t0 = time.time()
    model = build_sam3_video_model(device=device)
    model.eval()
    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # ── Read raw frames for visualization ──────────────────────────────────
    print(f"\n[2/4] Reading video frames: {args.video}")
    raw_frames, fps = read_video_frames(args.video, max_frames=args.max_frames)
    n_frames = len(raw_frames)
    if n_frames == 0:
        print("Error: no frames read from video")
        sys.exit(1)
    h, w = raw_frames[0].shape[:2]
    print(f"       {n_frames} frames @ {fps:.1f} FPS, resolution: {w}x{h}")

    # ── Initialize inference state ─────────────────────────────────────────
    print(f"\n[3/4] Initializing inference state ...")
    t0 = time.time()
    inference_state = model.init_state(
        resource_path=args.video,
        offload_video_to_cpu=False,
    )
    print(f"       State initialized in {time.time() - t0:.1f}s")

    # ── Add text prompt and propagate ──────────────────────────────────────
    prompt_text = " + ".join(args.prompt)
    print(f"\n[4/4] Running tracking with prompt: \"{prompt_text}\" ...")

    # Add prompt on frame 0 (the first prompt sets the text for detection)
    for prompt in args.prompt:
        print(f'  Adding prompt: "{prompt}"')
        model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )

    # Propagate across the video
    max_colors = 50
    colors = generate_distinct_colors(max_colors)
    overlay_frames = []
    total_objects = 0

    t0 = time.time()
    for frame_idx, outputs in model.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=args.max_frames if args.max_frames > 0 else None,
        reverse=False,
    ):
        obj_ids = outputs["out_obj_ids"]
        scores = outputs["out_probs"]
        masks = outputs["out_binary_masks"]  # (N, H_video, W_video) bool
        n_objects = len(obj_ids)
        total_objects = max(total_objects, n_objects)

        # Use the corresponding raw frame for visualization
        if frame_idx < len(raw_frames):
            raw_frame = raw_frames[frame_idx]
            # Resize masks to match raw frame if needed
            if masks.shape[0] > 0 and (masks.shape[1] != h or masks.shape[2] != w):
                masks_resized = np.zeros((masks.shape[0], h, w), dtype=bool)
                for i in range(masks.shape[0]):
                    masks_resized[i] = cv2.resize(
                        masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks = masks_resized

            overlay = overlay_masks_on_frame(raw_frame, masks, obj_ids, scores, colors)
            overlay_frames.append(overlay)

            # Show real-time display
            if args.show:
                cv2.imshow("SAM3 Video Tracking", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  [User interrupted]")
                    break

        # Print progress periodically
        if frame_idx % 30 == 0 or frame_idx == n_frames - 1:
            elapsed = time.time() - t0
            fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Frame {frame_idx + 1:>5d}/{n_frames} | "
                f"{n_objects} object(s) | "
                f"{fps_proc:.1f} FPS"
            )

    elapsed = time.time() - t0
    print(f"\n  Tracking completed in {elapsed:.1f}s ({len(overlay_frames)} frames processed)")

    if args.show:
        cv2.destroyAllWindows()

    # ── Save results ──────────────────────────────────────────────────────
    safe_prompt = "_".join(p.replace(" ", "_").replace("/", "_") for p in args.prompt)

    if args.save_video or (not args.save_frames and not args.show):
        # Default: save output video
        video_path = os.path.join(args.output, f"video_{safe_prompt}.mp4")
        print(f"\n  Saving output video: {video_path}")
        write_video(overlay_frames, video_path, fps)
        print(f"  -> saved {len(overlay_frames)} frames to {video_path}")

    if args.save_frames:
        frames_dir = os.path.join(args.output, f"frames_{safe_prompt}")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"\n  Saving individual frames to: {frames_dir}/")
        for i, frame in enumerate(overlay_frames):
            frame_path = os.path.join(frames_dir, f"{i:06d}.png")
            cv2.imwrite(frame_path, frame)
        print(f"  -> saved {len(overlay_frames)} frame(s)")

    # Save first and last frame as overview
    if len(overlay_frames) > 0:
        first_path = os.path.join(args.output, f"video_{safe_prompt}_first.png")
        save_overview_figure(raw_frames[0], overlay_frames[0], 0, prompt_text, total_objects, first_path)
        print(f"  -> first frame saved to {first_path}")

        if len(overlay_frames) > 1:
            last_idx = len(overlay_frames) - 1
            last_path = os.path.join(args.output, f"video_{safe_prompt}_last.png")
            save_overview_figure(
                raw_frames[min(last_idx, len(raw_frames) - 1)],
                overlay_frames[last_idx], last_idx, prompt_text, total_objects, last_path,
            )
            print(f"  -> last frame saved to {last_path}")

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"All results saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

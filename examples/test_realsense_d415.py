"""
SAM3 RealSense D415 Real-time Segmentation Demo

Captures live RGB frames from an Intel RealSense D415 camera and runs
SAM3 segmentation + tracking in real time.

Requirements:
    pip install pyrealsense2

Usage:
    # Basic: track "person" in real-time from RealSense D415
    python examples/test_realsense_d415.py

    # Custom prompt
    python examples/test_realsense_d415.py --prompt "only bottled water located on the conveyor belt"

    # Multiple prompts
    python examples/test_realsense_d415.py --prompt "person" "cup" "keyboard"

    # Set camera resolution (must be supported by D415)
    python examples/test_realsense_d415.py --width 1280 --height 720

    # Lower resolution for faster processing
    python examples/test_realsense_d415.py --width 640 --height 480

    # Record the output to an MP4 file
    python examples/test_realsense_d415.py --prompt "person" --record output.mp4

    # Save snapshots every N frames
    python examples/test_realsense_d415.py --prompt "person" --snapshot-interval 30

    # Use depth colormap overlay
    python examples/test_realsense_d415.py --prompt "person" --show-depth
"""

import argparse
import os
import signal
import sys
import time

import cv2
import matplotlib
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 is required. Install it with: pip install pyrealsense2")
    sys.exit(1)

# ─── SAM3 imports ───────────────────────────────────────────────────────────────
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    colors: list,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay binary masks and bounding boxes on a BGR frame.

    Args:
        frame_bgr:  (H, W, 3) uint8 BGR frame
        masks:      tensor of shape (N, 1, H, W) bool masks
        boxes:      tensor of shape (N, 4) bounding boxes [x1, y1, x2, y2]
        scores:     tensor of shape (N,) confidence scores
        colors:     list of RGB tuples
        alpha:      mask transparency

    Returns:
        (H, W, 3) uint8 BGR frame with overlays
    """
    vis = frame_bgr.copy()
    n_objects = len(scores)

    for i in range(n_objects):
        mask_np = masks[i].squeeze(0).cpu().numpy()
        box_np = boxes[i].cpu().numpy()
        score = scores[i].item()
        color_idx = i % len(colors)
        rgb = np.array(colors[color_idx]) * 255

        # Resize mask to frame size if needed
        h, w = vis.shape[:2]
        if mask_np.shape[0] != h or mask_np.shape[1] != w:
            mask_np = cv2.resize(
                mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Overlay mask
        colored_mask = np.zeros_like(vis, dtype=np.float32)
        colored_mask[..., 0] = rgb[2]  # BGR order
        colored_mask[..., 1] = rgb[1]
        colored_mask[..., 2] = rgb[0]
        mask_3d = mask_np[:, :, np.newaxis].astype(np.float32)
        vis = (vis.astype(np.float32) * (1 - mask_3d * alpha) + colored_mask * mask_3d * alpha)
        vis = vis.astype(np.uint8)

        # Draw bounding box
        x1, y1, x2, y2 = int(box_np[0]), int(box_np[1]), int(box_np[2]), int(box_np[3])
        bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_color, 2)
        label = f"#{i} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), bgr_color, -1)
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def draw_hud(frame: np.ndarray, fps: float, n_objects: int, prompt_text: str) -> np.ndarray:
    """Draw a heads-up display with FPS and status info."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Semi-transparent top bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # Status text
    status = f'SAM3 RealSense D415 | Prompt: "{prompt_text}" | {n_objects} obj | {fps:.1f} FPS'
    cv2.putText(
        vis, status, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
    )

    # Hotkeys
    hotkey_text = "[Q] Quit  [S] Snapshot  [P] Change prompt  [D] Toggle depth"
    cv2.putText(
        vis, hotkey_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return vis


# ─── RealSense setup ───────────────────────────────────────────────────────────
def setup_realsense(width: int, height: int, fps: int, enable_depth: bool) -> tuple:
    """
    Configure and start the RealSense D415 pipeline.

    Returns:
        pipeline: rs.pipeline
        align:    rs.align (to align depth to color)
        profile:  rs.pipeline_profile
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Enable depth stream if requested
    if enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Try to start the pipeline
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error: Cannot start RealSense pipeline: {e}")
        print("\nTips:")
        print("  - Make sure the RealSense D415 is connected via USB 3.0")
        print("  - Try a different resolution: --width 640 --height 480")
        print("  - Check: realsense-viewer")
        sys.exit(1)

    # Get device info
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"  Camera: {device_name} (S/N: {serial})")

    # Align depth to color
    align = rs.align(rs.stream.color) if enable_depth else None

    return pipeline, align, profile


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 RealSense D415 Real-time Segmentation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, nargs="+", default=["person"],
        help='Text prompt(s) for segmentation, e.g. --prompt "hand" "cup"',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Camera resolution width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Camera resolution height (default: 720)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Camera FPS (default: 30)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(sam3_root, "examples", "output"),
        help="Output directory for snapshots/recordings (default: examples/output/)",
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Path to save output MP4 recording (e.g. output.mp4)",
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=0,
        help="Auto-save snapshot every N frames (0 = disabled)",
    )
    parser.add_argument(
        "--show-depth", action="store_true",
        help="Show depth colormap alongside the segmentation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
    print("SAM3 RealSense D415 Real-time Segmentation Demo")
    print("=" * 60)

    print(f"\n[1/3] Loading SAM3 model (device={device}) ...")
    t0 = time.time()
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)
    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # ── Setup RealSense ────────────────────────────────────────────────────
    print(f"\n[2/3] Connecting to RealSense D415 ({args.width}x{args.height} @ {args.fps}fps) ...")
    pipeline, align, profile = setup_realsense(args.width, args.height, args.fps, args.show_depth)
    print(f"       Camera ready!")

    # ── Prepare visualization ──────────────────────────────────────────────
    max_colors = 30
    colors = generate_distinct_colors(max_colors)
    prompt_text = " + ".join(args.prompt)
    current_prompts = list(args.prompt)

    # Video writer for recording
    writer = None
    if args.record:
        record_path = args.record if os.path.isabs(args.record) else os.path.join(args.output, args.record)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record_path, fourcc, args.fps, (args.width, args.height))
        print(f"  Recording to: {record_path}")

    # ── Main loop ──────────────────────────────────────────────────────────
    print(f'\n[3/3] Starting real-time segmentation with prompt: "{prompt_text}"')
    print("       Press Q to quit, S for snapshot, P to change prompt, D to toggle depth\n")

    frame_count = 0
    fps_display = 0.0
    fps_timer = time.time()
    fps_frame_count = 0
    show_depth = args.show_depth
    running = True

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            # ── Capture frame ──────────────────────────────────────────────
            frameset = pipeline.wait_for_frames()

            if align and show_depth:
                frameset = align.process(frameset)

            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())

            # ── Get depth frame if enabled ─────────────────────────────────
            depth_colormap = None
            if show_depth:
                depth_frame = frameset.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET,
                    )

            # ── Run SAM3 segmentation ──────────────────────────────────────
            from PIL import Image
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inference_state = processor.set_image(pil_image)

            # Run segmentation for each prompt and combine results
            all_masks = []
            all_boxes = []
            all_scores = []

            for prompt in current_prompts:
                processor.reset_all_prompts(inference_state)
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)

                if len(output["scores"]) > 0:
                    all_masks.append(output["masks"])
                    all_boxes.append(output["boxes"])
                    all_scores.append(output["scores"])

            # Combine results from all prompts
            if all_masks:
                combined_masks = torch.cat(all_masks, dim=0)
                combined_boxes = torch.cat(all_boxes, dim=0)
                combined_scores = torch.cat(all_scores, dim=0)
                n_objects = len(combined_scores)

                overlay = overlay_masks_on_frame(
                    frame_bgr, combined_masks, combined_boxes, combined_scores, colors,
                )
            else:
                overlay = frame_bgr.copy()
                n_objects = 0

            # ── FPS calculation ────────────────────────────────────────────
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # ── Draw HUD ───────────────────────────────────────────────────
            display = draw_hud(overlay, fps_display, n_objects, prompt_text)

            # ── Compose final display ──────────────────────────────────────
            if show_depth and depth_colormap is not None:
                # Side-by-side: segmentation | depth
                depth_resized = cv2.resize(depth_colormap, (display.shape[1], display.shape[0]))
                display = np.hstack([display, depth_resized])

            cv2.imshow("SAM3 RealSense D415", display)

            # ── Record if enabled ──────────────────────────────────────────
            if writer:
                writer.write(overlay)

            # ── Auto-snapshot ──────────────────────────────────────────────
            if args.snapshot_interval > 0 and frame_count % args.snapshot_interval == 0:
                snap_path = os.path.join(args.output, f"snapshot_{frame_count:06d}.png")
                cv2.imwrite(snap_path, overlay)

            frame_count += 1

            # ── Handle keyboard input ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                print("\n[Quit]")
                break

            elif key == ord("s"):  # Snapshot
                snap_path = os.path.join(args.output, f"snapshot_{frame_count:06d}.png")
                cv2.imwrite(snap_path, overlay)
                print(f"  Snapshot saved: {snap_path}")

            elif key == ord("p"):  # Change prompt
                cv2.destroyAllWindows()
                new_prompt = input("Enter new prompt(s) (comma-separated): ").strip()
                if new_prompt:
                    current_prompts = [p.strip() for p in new_prompt.split(",") if p.strip()]
                    prompt_text = " + ".join(current_prompts)
                    print(f'  New prompt: "{prompt_text}"')

            elif key == ord("d"):  # Toggle depth
                show_depth = not show_depth
                print(f"  Depth display: {'ON' if show_depth else 'OFF'}")

    finally:
        # ── Cleanup ────────────────────────────────────────────────────────
        print("\nShutting down ...")
        pipeline.stop()
        if writer:
            writer.release()
            print(f"  Recording saved.")
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames.")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

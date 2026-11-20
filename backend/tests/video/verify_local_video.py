"""Run the video pipeline against a user-provided local video file."""

import argparse
import math
from pathlib import Path
import sys

import cv2

from app.modules.video.preprocessing import validate_video_file
from app.modules.video.video_analysis import process_video


def _read_source_timestamps(
    video_path: Path, frame_indices: set[int]
) -> dict[int, float]:
    """Read source timestamps independently for selected frame indices."""
    capture = cv2.VideoCapture(str(video_path))
    timestamps: dict[int, float] = {}
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not reopen the video to verify timestamps.")
        final_index = max(frame_indices, default=-1)
        for frame_index in range(final_index + 1):
            success, _ = capture.read()
            if not success:
                break
            if frame_index in frame_indices:
                timestamps[frame_index] = (
                    capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )
    finally:
        capture.release()

    if timestamps.keys() != frame_indices:
        raise ValueError("Could not reread all extracted frames to verify timestamps.")
    return timestamps


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line options for repeatable local-video verification."""
    parser = argparse.ArgumentParser(
        description="Validate a local video, inspect metadata, and extract frames."
    )
    parser.add_argument("video_path", type=Path, help="Path to a local video file")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where extracted JPEG frames will be saved",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Frame sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of extracted frames (default: 10)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="Optional maximum video size in MiB",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify a local video and return a process exit code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    max_size_bytes: int | None = None
    if arguments.max_size_mb is not None:
        if arguments.max_size_mb <= 0:
            parser.error("--max-size-mb must be greater than zero.")
        max_size_bytes = round(arguments.max_size_mb * 1024 * 1024)

    validation = validate_video_file(
        arguments.video_path, max_size_bytes=max_size_bytes
    )
    if not validation.valid:
        print("Video validation failed:")
        for error in validation.errors:
            print(f"- {error}")
        return 2

    try:
        result = process_video(
            arguments.video_path,
            output_directory=arguments.output_dir,
            interval_seconds=arguments.interval,
            max_frames=arguments.max_frames,
            max_size_bytes=max_size_bytes,
        )
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        print(f"Video processing failed: {error}", file=sys.stderr)
        return 1

    frames = result.extraction.frames
    if len(frames) > arguments.max_frames:
        print("Verification failed: extracted frame limit was exceeded.", file=sys.stderr)
        return 1

    try:
        source_timestamps = _read_source_timestamps(
            arguments.video_path, {frame.frame_index for frame in frames}
        )
    except (cv2.error, OSError, ValueError) as error:
        print(f"Timestamp verification failed: {error}", file=sys.stderr)
        return 1

    for frame_position, frame in enumerate(frames):
        decoded_image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if decoded_image is None:
            print(
                f"Verification failed: saved JPEG is unreadable: {frame.path}",
                file=sys.stderr,
            )
            return 1

        decoded_height, decoded_width = decoded_image.shape[:2]
        if (decoded_width, decoded_height) != (frame.width, frame.height):
            print(
                f"Verification failed: saved JPEG dimensions differ from frame "
                f"{frame.frame_index}.",
                file=sys.stderr,
            )
            return 1
        source_timestamp = source_timestamps[frame.frame_index]
        if (
            not math.isfinite(source_timestamp)
            or abs(frame.timestamp_seconds - source_timestamp) > 0.001
        ):
            print(
                f"Verification failed: timestamp differs from source for frame "
                f"{frame.frame_index}.",
                file=sys.stderr,
            )
            return 1
        if frame_position:
            previous_frame = frames[frame_position - 1]
            actual_interval = frame.timestamp_seconds - previous_frame.timestamp_seconds
            minimum_interval = arguments.interval - 1 / (result.metadata.fps or 1)
            if actual_interval < minimum_interval:
                print(
                    "Verification failed: extracted timestamps are closer together "
                    "than the requested sampling interval allows.",
                    file=sys.stderr,
                )
                return 1

    print("Video processing succeeded.")
    print(f"Input: {result.extraction.video_path}")
    print(f"Size: {result.metadata.size_bytes} bytes")
    print(f"Duration: {result.metadata.duration_seconds} seconds")
    print(f"FPS: {result.metadata.fps}")
    print(f"Frame count: {result.metadata.frame_count}")
    print(f"Resolution: {result.metadata.resolution}")
    print(f"Aspect ratio: {result.metadata.aspect_ratio}")
    print(f"Sampling interval: {result.extraction.interval_seconds} seconds")
    print(f"Output directory: {result.extraction.output_directory}")
    print(f"Extracted frames: {len(frames)} (maximum {arguments.max_frames})")

    for frame in frames:
        print(
            f"- index={frame.frame_index}, "
            f"timestamp={frame.timestamp_seconds:.6f}s, "
            f"dimensions={frame.width}x{frame.height}, "
            f"jpeg_readable=yes, path={frame.path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

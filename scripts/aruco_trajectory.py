"""
Generate mapping_camera_trajectory.csv from ArUco tag detections.
Bypasses SLAM by using the ArUco tag as the world coordinate origin.

Usage:
    uv run python scripts/aruco_trajectory.py \
        --video data/demos/mapping/converted_60fps_raw_video.mp4 \
        --intrinsics packages/umi/defaults/calibration/gopro13_intrinsics_2_7k_C2.json \
        --aruco-config packages/umi/defaults/calibration/aruco_config.yaml \
        --tag-id 13 \
        --output data/demos/mapping/mapping_camera_trajectory.csv
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/umi/src"))
from umi.common.cv_util import (
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def aruco_to_camera_pose(rvec, tvec):
    """
    Convert solvePnP output to camera pose in tag (world) frame.

    solvePnP gives the transform: X_cam = R @ X_tag + t
    So camera position in tag frame = -R^T @ t
    Camera orientation in tag frame = R^T
    """
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).squeeze()
    cam_rot = Rotation.from_matrix(R.T)
    return cam_pos, cam_rot


def main():
    parser = argparse.ArgumentParser(description="Generate camera trajectory from ArUco detections")
    parser.add_argument("--video", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-config", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keyframe-interval", type=int, default=5,
                        help="Mark every N-th detected frame as keyframe (default: 5)")
    args = parser.parse_args()

    # Load intrinsics
    with open(args.intrinsics) as f:
        intrinsics = parse_fisheye_intrinsics(json.load(f))

    # Load ArUco config
    with open(args.aruco_config) as f:
        aruco_cfg = yaml.safe_load(f)
    aruco_result = parse_aruco_config(aruco_cfg)
    aruco_dict = aruco_result["aruco_dict"]
    marker_size_map = aruco_result["marker_size_map"]

    tag_id = args.tag_id
    tag_size = marker_size_map.get(tag_id, 0.16)
    print(f"Reference tag ID: {tag_id}, size: {tag_size} m")

    rows = []
    detected_count = 0
    keyframe_counter = 0

    with av.open(args.video) as container:
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames
        fps = float(video_stream.average_rate)
        print(f"Video: {total_frames} frames at {fps:.1f} fps")

        for frame_idx, frame in enumerate(tqdm(container.decode(video=0), total=total_frames)):
            timestamp = float(frame.pts * video_stream.time_base)
            img = frame.to_ndarray(format="bgr24")

            tag_dict = detect_localize_aruco_tags(img, aruco_dict, marker_size_map, intrinsics)

            if tag_id not in tag_dict:
                rows.append({
                    "frame_id": frame_idx,
                    "timestamp": timestamp,
                    "is_lost": True,
                    "is_keyframe": False,
                    "x": np.nan, "y": np.nan, "z": np.nan,
                    "q_x": np.nan, "q_y": np.nan, "q_z": np.nan, "q_w": np.nan,
                })
                continue

            det = tag_dict[tag_id]
            cam_pos, cam_rot = aruco_to_camera_pose(det["rvec"], det["tvec"])
            quat = cam_rot.as_quat()  # [x, y, z, w]

            detected_count += 1
            keyframe_counter += 1
            is_keyframe = (keyframe_counter % args.keyframe_interval == 0)

            rows.append({
                "frame_id": frame_idx,
                "timestamp": timestamp,
                "is_lost": False,
                "is_keyframe": is_keyframe,
                "x": float(cam_pos[0]),
                "y": float(cam_pos[1]),
                "z": float(cam_pos[2]),
                "q_x": float(quat[0]),
                "q_y": float(quat[1]),
                "q_z": float(quat[2]),
                "q_w": float(quat[3]),
            })

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(rows)
    keyframes = df["is_keyframe"].sum()
    print(f"\nResults:")
    print(f"  Total frames:     {total}")
    print(f"  Tag detected:     {detected_count} ({detected_count/total*100:.1f}%)")
    print(f"  Keyframes:        {keyframes}")
    print(f"  Saved to:         {output_path}")

    if detected_count == 0:
        print("\nWARNING: Tag not detected in any frame. Check that:")
        print(f"  - Tag ID {tag_id} is visible in the mapping video")
        print(f"  - The intrinsics file matches the camera")


if __name__ == "__main__":
    main()

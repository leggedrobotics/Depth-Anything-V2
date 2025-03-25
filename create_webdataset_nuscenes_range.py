import os
import io
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from webdataset import ShardWriter
from multiprocessing import Pool, cpu_count
import random

from nuscenes.utils.data_classes import LidarPointCloud


class RangeProjection:
    def __init__(self, fov_up, fov_down, proj_w, proj_h, fov_left=-180, fov_right=180):
        self.fov_up = np.deg2rad(fov_up)
        self.fov_down = np.deg2rad(fov_down)
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.fov_left = np.deg2rad(fov_left)
        self.fov_right = np.deg2rad(fov_right)
        self.fov_h = abs(self.fov_left) + abs(self.fov_right)

        self.proj_w = proj_w
        self.proj_h = proj_h

    def do_projection(self, pointcloud: np.ndarray):
        depth = np.linalg.norm(pointcloud[:, :3], axis=1)
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]

        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_x = (yaw + abs(self.fov_left)) / self.fov_h
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v

        proj_x = np.clip(np.floor(proj_x * self.proj_w), 0, self.proj_w - 1).astype(np.int32)
        proj_y = np.clip(np.floor(proj_y * self.proj_h), 0, self.proj_h - 1).astype(np.int32)

        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_pointcloud = np.full((self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)

        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth, indices, pointcloud = depth[order], indices[order], pointcloud[order]
        proj_y, proj_x = proj_y[order], proj_x[order]

        proj_range[proj_y, proj_x] = depth
        proj_pointcloud[proj_y, proj_x] = pointcloud
        proj_idx[proj_y, proj_x] = indices

        return proj_pointcloud, proj_range


def process_file(args):
    idx, file_path = args
    try:
        pc = LidarPointCloud.from_file(file_path)
        points = pc.points.T  # (N, 4)

        projection = RangeProjection(fov_up=10, fov_down=-30, proj_w=2048, proj_h=32)
        proj_pc, proj_range = projection.do_projection(points)
        proj_intensity = proj_pc[..., 3][..., np.newaxis]  # (H, W, 1)

        proj_feature = np.concatenate([
            proj_range[..., np.newaxis],  # (H, W, 1)
            proj_pc[..., :3],             # (H, W, 3)
            proj_intensity                # (H, W, 1)
        ], axis=-1)  # Final shape: (H, W, 5)

        npz_buf = io.BytesIO()
        np.savez_compressed(npz_buf, data=proj_feature)

        fname = Path(file_path).stem.replace(".pcd", "")

        sample = {
            "__key__": fname,
            "npz": npz_buf.getvalue(),
            "json": json.dumps({
                "shape": proj_feature.shape,
                "filename": os.path.basename(file_path),
                "frame_id": idx,
                "dataset": "nuscenes",
                "fields": ["range", "x", "y", "z", "intensity"]
            }).encode("utf-8")
        }
        return sample

    except Exception as e:
        print(f"⚠️ Skipping {file_path} due to error: {e}")
        return None


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    lidar_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir) if f.endswith(".bin")
    ]
    
    # Shuffle the lidar files
    random.shuffle(lidar_files)

    print(f"📦 Found {len(lidar_files)} .bin files")

    shard_pattern = os.path.join(args.output_dir, "nuscenes-range-shard-%05d.tar")

    with ShardWriter(shard_pattern, maxcount=args.shard_size) as sink:
        with Pool(processes=args.num_workers) as pool:
            for sample in tqdm(pool.imap_unordered(process_file, enumerate(lidar_files)), total=len(lidar_files)):
                if sample is not None:
                    sink.write(sample)

    print("✅ WebDataset shards created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Path to LIDAR_TOP folder with .bin files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for WebDataset shards")
    parser.add_argument("--shard-size", type=int, default=3300, help="Max samples per shard")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel processes (default: all cores)")
    args = parser.parse_args()
    main(args)

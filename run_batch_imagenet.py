import os
import time
import numpy as np
import torch
import webdataset as wds
import cv2
import argparse
import tarfile
import tempfile
from itertools import islice
from torchvision import transforms
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import json
from tqdm import tqdm

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description="Convert ImageNet-22K to Depth Maps (WebDataset)")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for WebDataset shards")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
parser.add_argument("--input-size", type=int, default=518, help="Input size for DepthAnythingV2")
parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
parser.add_argument("--start-split", type=int, default=0, help="Starting shard idx")
parser.add_argument("--num-splits", type=int, default=4, help="Num of splits of total shards, for parallel processing")
args = parser.parse_args()

# ========== Device Selection ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# ========== Load DepthAnythingV2 Model ==========
model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

print(f"📥 Loading {args.encoder} model...")
depth_model = DepthAnythingV2(**model_configs[args.encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location="cpu"))
depth_model = depth_model.to(DEVICE).eval()

# ========== Hugging Face WebDataset ==========
fs = HfFileSystem()
splits = {'train': '**/*-train-*.tar'}
token = get_token()
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/timm/imagenet-22k-wds/" + splits["train"])]

# ========== Split Files for Parallel Processing ==========
total_files = len(files)
split_size = total_files // args.num_splits
start_idx = args.start_split * split_size
end_idx = (args.start_split + 1) * split_size if args.start_split + 1 < args.num_splits else total_files
files = files[start_idx:end_idx]

# ========== Output Directory ==========
os.makedirs(args.output_dir, exist_ok=True)

# ========== Colormap for Visualization ==========
cmap = plt.get_cmap("magma")

# ========== Transformations ==========
def preprocess_image(image):
    """Apply standard preprocessing before depth inference"""
    transform = transforms.Compose([
        Resize(
            width=args.input_size,
            height=args.input_size,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    img_array = np.array(image)  # Convert PIL Image to numpy
    return transform({'image': img_array})['image']

# ========== Helper Functions ==========
def normalize_depth(depth):
    """Normalize depth map to [0, 255] for visualization"""
    depth_np = depth.cpu().numpy()
    return ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)

def save_shard(data, shard_index):
    """Save a shard as a .tar file"""
    shard_path = os.path.join(args.output_dir, f"shard-{shard_index:05d}.tar")
    
    with tarfile.open(shard_path, "w") as tar:
        for idx, (img_name, depth_bytes, cls) in enumerate(data):
            # Remove extension from image name
            img_name = os.path.splitext(img_name)[0]
            
            # Save depth image as PNG inside the tar file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_depth:
                tmp_depth.write(depth_bytes)
                tmp_depth.close()
                tar.add(tmp_depth.name, arcname=f"{img_name}.png")
                os.unlink(tmp_depth.name)

            # Save class metadata as JSON inside the tar file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_cls:  # Open in text mode ('w')
                json.dump(cls, tmp_cls)  # Directly dump cls (not wrapping in another dict)
                tmp_cls.close()
                tar.add(tmp_cls.name, arcname=f"{img_name}.json")
                os.unlink(tmp_cls.name)

    print(f"✅ Saved {shard_path}")

# ========== Processing Loop ==========
shard_data = []
shard_index = 0
batch = []


for i, file in enumerate(files):
    print(f"Processing file {i + start_idx}: {file}")
    shard_index = i + start_idx
    urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset")]
    urls = f"pipe: curl --connect-timeout 60 --retry 10 --retry-delay 5 -f -s -L -H 'Authorization: Bearer {token}' {'::'.join(urls)}"
    dataset = wds.WebDataset(urls).decode("rgb").to_tuple("__key__", "jpg", "json")

    with torch.no_grad():
        start_time = time.time()
        for idx, (key, image, cls) in enumerate(tqdm(dataset, desc="Processing images", total=3339)):
            # print(f"Image Name: {key}.jpg, Class: {cls}")
            # print(f"Processing {idx + 1}...")
            # Apply transforms
            img_tensor = preprocess_image(image)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(DEVICE)
            batch.append((img_tensor, image, cls, key))
            
            if len(batch) == args.batch_size:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
                # Stack images and process depth
                images = torch.cat([b[0] for b in batch], dim=0)
                patch_h, patch_w = images.shape[-2] // 14, images.shape[-1] // 14

                features = depth_model.pretrained.get_intermediate_layers(
                    images, 
                    depth_model.intermediate_layer_idx[depth_model.encoder], 
                    return_class_token=True
                )
                depths = depth_model.depth_head(features, patch_h, patch_w)
                depths = torch.nn.functional.relu(depths)

                end_event.record()
                torch.cuda.synchronize()
                gpu_time = start_event.elapsed_time(end_event)

                # Process each image in batch
                for i, (_, orig_img, cls, key) in enumerate(batch):
                    depth = depths[i]
                    depth_resized = torch.nn.functional.interpolate(
                        depth.unsqueeze(1), 
                        size=(orig_img.shape[0], orig_img.shape[1]), 
                        mode="bilinear",
                        align_corners=True
                    )[0, 0]

                    depth_np = normalize_depth(depth_resized)

                    # Encode to PNG bytes
                    _, depth_encoded = cv2.imencode(".png", depth_np)

                    shard_data.append((key, depth_encoded.tobytes(), cls))

                batch = []  # Reset batch           
                
    # Save shard if limit reached
    save_shard(shard_data, shard_index)
    shard_data = []  # Reset buffer

    # ========== Final Output ==========
    total_time = time.time() - start_time
    print(f"Time per Shard: {total_time:.2f}s, Avg time per image: {total_time / idx:.3f}s")

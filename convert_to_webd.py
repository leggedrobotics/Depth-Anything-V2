import os
import json
import tarfile
import io
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# === CONFIGURATION ===
ROOT_DIR = "/media/patelm/ssd/imagenet-1k/val"  # Set the path to the folder containing class subdirectories
OUTPUT_DIR = "/media/patelm/ssd/imagenet-1k/val_webd"  # Output directory for WebDataset
SHARD_SIZE = 1000  # Number of samples per shard
NUM_WORKERS = min(cpu_count(), 8)  # Use up to 8 CPU cores

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all images and labels
image_list = []
for class_name in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-directory files

    for img_file in os.listdir(class_path):
        if img_file.endswith(".png"):
            img_path = os.path.join(class_path, img_file)
            image_list.append((img_path, class_name, img_file))

# Shuffle to ensure proper distribution across shards
np.random.shuffle(image_list)

# Compute number of shards
num_shards = len(image_list) // SHARD_SIZE + int(len(image_list) % SHARD_SIZE > 0)
print(f"Creating {num_shards} shards with multiprocessing using {NUM_WORKERS} workers...")


def process_shard(shard_id):
    """Function to process a single shard."""
    shard_path = os.path.join(OUTPUT_DIR, f"dataset-{shard_id:05d}.tar")
    if os.path.exists(shard_path):  # Skip already processed shards
        print(f"Skipping existing shard: {shard_path}")
        return

    with tarfile.open(shard_path, "w") as tar:
        for idx in range(SHARD_SIZE):
            global_idx = shard_id * SHARD_SIZE + idx
            if global_idx >= len(image_list):
                break
            
            img_path, class_name, img_name = image_list[global_idx]

            try:
                # Read image bytes directly
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                # Save image as .png inside tar
                img_tar_name = f"{class_name}_{img_name}"
                img_info = tarfile.TarInfo(img_tar_name)
                img_info.size = len(img_bytes)
                tar.addfile(img_info, io.BytesIO(img_bytes))

                # Save metadata as .json
                metadata = {"class_name": class_name}
                json_bytes = json.dumps(metadata).encode("utf-8")

                img_name_noext = os.path.splitext(img_name)[0] 
                json_name = f"{class_name}_{img_name_noext}.json"
                json_info = tarfile.TarInfo(json_name)
                json_info.size = len(json_bytes)
                tar.addfile(json_info, io.BytesIO(json_bytes))

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue

    print(f"✅ Finished processing shard {shard_id}: {shard_path}")


# Run multiprocessing
with Pool(NUM_WORKERS) as pool:
    list(tqdm(pool.imap_unordered(process_shard, range(num_shards)), total=num_shards))

print(f"✅ WebDataset conversion completed successfully! WebDataset stored in {OUTPUT_DIR}")

import os
import tarfile
import argparse
import io
import json
from PIL import Image
from glob import glob
from tqdm import tqdm
from webdataset import ShardWriter
from pathlib import Path

def extract_pngs_from_tar(tar_path, keyword):
    images = []
    tar_root_name = Path(tar_path).stem

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith(".png"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            # Read PNG bytes
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            meta = {
                "filename": os.path.basename(member.name),
                "width": width,
                "height": height,
                "bit_depth": "16-bit",
                "depth_resolution":  512.0,
                "root_name": tar_root_name,
                "dataset": keyword,
            }

            key = os.path.splitext(os.path.basename(member.name))[0]

            images.append({
                "__key__": key,
                "png": img_bytes,
                "json": json.dumps(meta).encode("utf-8"),
            })

    return images

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ✅ Filter tar files by keyword in FILENAME only (not inside path)
    tar_files = [
        t for t in glob(os.path.join(args.input_dir, "**/*.tar"), recursive=True)
        if args.keyword in os.path.basename(t)
    ]

    print(f"🔍 Found {len(tar_files)} tar files matching keyword '{args.keyword}'")

    shard_pattern = os.path.join(args.output_dir, f"{args.keyword}-depth-shard-%05d.tar")

    sample_count = 0
    shard_count = 0

    with ShardWriter(shard_pattern, maxcount=args.shard_size) as sink:
        for tar_path in tqdm(tar_files, desc="Processing TAR files"):
            images = extract_pngs_from_tar(tar_path, args.keyword)  # No change here
            for sample in images:
                sink.write(sample)
                sample_count += 1

    print(f"✅ Done! Total samples written: {sample_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with TAR files")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write WebDataset shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="Max samples per shard")
    parser.add_argument("--keyword", type=str, required=True, help="Keyword to match PNG file paths inside tars")
    args = parser.parse_args()
    main(args)

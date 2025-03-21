import os
import zipfile
import argparse
import io
import json
from PIL import Image
from glob import glob
from tqdm import tqdm
from webdataset import ShardWriter
from pathlib import Path

def extract_pngs_from_zip(zip_path, fps_scale, keyword):
    images = []
    zip_root_name = Path(zip_path).stem

    with zipfile.ZipFile(zip_path, 'r') as archive:
        # Filter PNGs and sort
        png_members = sorted(
            [m for m in archive.namelist() if m.lower().endswith('.png')]
        )

        for idx, member in enumerate(png_members):
            if idx % fps_scale != 0:
                continue

            try:
                img_bytes = archive.read(member)
                img = Image.open(io.BytesIO(img_bytes))
                img.load()  # force-load image to verify

                width, height = img.size

                meta = {
                    "filename": os.path.basename(member),
                    "width": width,
                    "height": height,
                    "bit_depth": "16-bit",
                    "depth_resolution": 512.0,
                    "root_name": zip_root_name,
                    "dataset": keyword,
                }

                filename = os.path.basename(member)
                key = filename.replace(".png", "").replace(".", "_")  # sanitize: replace dots except extension


                images.append({
                    "__key__": key,
                    "png": img_bytes,
                    "json": json.dumps(meta).encode("utf-8"),
                })

            except Exception as e:
                print(f"⚠️ Skipping '{member}' in '{zip_path}' due to error: {e}")
                continue

    return images


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    zip_files = glob(os.path.join(args.input_dir, "**/*.zip"), recursive=True)
    print(f"📦 Found {len(zip_files)} zip files")

    shard_pattern = os.path.join(args.output_dir, f"{args.keyword}-depth-shard-%05d.tar")

    sample_count = 0

    with ShardWriter(shard_pattern, maxcount=args.shard_size) as sink:
        for zip_path in tqdm(zip_files, desc="Processing ZIPs"):
            # if args.keyword not in os.path.basename(zip_path):
            #     continue

            images = extract_pngs_from_zip(zip_path, args.fps_scale, args.keyword)
            for sample in images:
                sink.write(sample)
                sample_count += 1

    print(f"✅ Done! Total samples written: {sample_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing ZIPs")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for WebDataset shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per shard")
    parser.add_argument("--fps-scale", type=int, default=12, help="Frame skip scale (e.g. 4 = 60fps to 15fps)")
    parser.add_argument("--keyword", type=str, required=True, help="Only process ZIPs with this keyword in filename")
    args = parser.parse_args()

    main(args)

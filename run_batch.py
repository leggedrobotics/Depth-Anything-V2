import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import time

from depth_anything_v2.dpt import DepthAnythingV2


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transforms=None, extensions=['jpg', 'jpeg', 'png', 'JPEG', 'JPG']):
        self.folder_path = folder_path
        self.transforms = transforms
        
        # Get all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True))
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Original image dimensions
        h, w = image.shape[:2]
        
        # Apply transforms if any
        if self.transforms:
            transformed = self.transforms({'image': image})
            image = transformed['image']
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        
        return {
            'image': image_tensor,
            'path': image_path,
            'original_size': (h, w)
        }


class BatchTransform:
    def __init__(self, input_size=518):
        self.input_size = input_size
        
    def __call__(self, image_dict):
        from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
        
        transform = Compose([
            Resize(
                width=self.input_size,
                height=self.input_size,
                resize_target=True,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        return transform(image_dict)


def collate_fn(batch):
    images = torch.cat([item['image'] for item in batch], dim=0)
    paths = [item['path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'paths': paths,
        'original_sizes': original_sizes
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Batch Processing')
    
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='./depth_results', help='Output directory for depth maps')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--input-size', type=int, default=448, help='Input image size')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Output grayscale depth instead of color')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only save depth prediction without original image')
    parser.add_argument('--save-npz', dest='save_npz', action='store_true', help='Save depth as npz file')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    print(f"Loading {args.encoder} model...")
    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    
    # Create colormap
    cmap = plt.get_cmap('Spectral_r')
    
    # Create dataset and dataloader
    transform = BatchTransform(input_size=args.input_size)
    dataset = ImageFolderDataset(args.input_dir, transforms=transform)
    
    if len(dataset) == 0:
        print(f"No images found in {args.input_dir}")
        exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Found {len(dataset)} images to process")
    

    # CUDA event-based timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0.0  # Total time tracking
    start_time = time.time()  # Wall time tracking
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_start = time.time()  # Start CPU timing
            start_event.record()  # Start CUDA event timing

            # Move images to device
            images = batch['images'].to(DEVICE, non_blocking=True)

            # Get patch dimensions
            patch_h, patch_w = images.shape[-2] // 14, images.shape[-1] // 14

            # Forward pass (Depth Estimation)
            features = depth_model.pretrained.get_intermediate_layers(
                images, 
                depth_model.intermediate_layer_idx[depth_model.encoder], 
                return_class_token=True
            )
            depths = depth_model.depth_head(features, patch_h, patch_w)
            depths = torch.nn.functional.relu(depths)

            end_event.record()  # Stop CUDA timing
            torch.cuda.synchronize()  # Ensure event is finished
            gpu_time = start_event.elapsed_time(end_event)  # GPU time in ms

            for j in range(len(batch['paths'])):
                img_path = batch['paths'][j]
                original_h, original_w = batch['original_sizes'][j]
                depth = depths[j]

                # Resize depth map to match original image dimensions
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1), 
                    (original_h, original_w), 
                    mode="bilinear", 
                    align_corners=True
                )[0, 0]

                # # Convert to numpy (Move to CPU asynchronously)
                # depth_np = depth.cpu().numpy()

                # # Save as .npz if required
                # if args.save_npz:
                #     np.savez_compressed(
                #         os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.npz'), 
                #         depth=np.uint16(depth_np)
                #     )

                # # Normalize depth for visualization
                # depth_viz = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
                # depth_viz = depth_viz.astype(np.uint8)

                # # Apply colormap unless grayscale is requested
                # if args.grayscale:
                #     depth_viz = np.repeat(depth_viz[..., np.newaxis], 3, axis=-1)
                # else:
                #     depth_viz = (plt.get_cmap("magma")(depth_viz)[:, :, :3] * 255).astype(np.uint8)

                # # Save output visualization
                # output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')

                # if args.pred_only:
                #     cv2.imwrite(output_path, depth_viz)
                # else:
                #     # Read original image for side-by-side comparison
                #     original_img = cv2.imread(img_path)

                #     # Resize original image if needed to match depth map height
                #     if original_img.shape[0] != depth_viz.shape[0]:
                #         scale = depth_viz.shape[0] / original_img.shape[0]
                #         new_width = int(original_img.shape[1] * scale)
                #         original_img = cv2.resize(original_img, (new_width, depth_viz.shape[0]), interpolation=cv2.INTER_AREA)

                #     # Create separator
                #     separator = np.ones((depth_viz.shape[0], 50, 3), dtype=np.uint8) * 255

                #     # Concatenate original and depth images
                #     combined = cv2.hconcat([original_img, separator, depth_viz])
                #     cv2.imwrite(output_path, combined)

            batch_time = time.time() - batch_start  # CPU batch time
            total_time += batch_time  # Accumulate total time

            processed = min((i + 1) * args.batch_size, len(dataset))
            print(f"Processed {processed}/{len(dataset)} images, "
                f"Batch {i+1}/{len(dataloader)} time: {batch_time:.3f}s, "
                f"GPU Time: {gpu_time:.2f}ms, "
                f"Avg per image: {batch_time / len(batch['paths']):.3f}s")

    elapsed = time.time() - start_time  # Total wall time
    print(f"✅ Processing complete! Total time: {elapsed:.3f}s, "
        f"Avg time per image: {total_time / len(dataset):.3f}s")
    print(f"Results saved to {args.output_dir}")
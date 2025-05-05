import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import traceback
import math
import time

from Unet.UNet_3Plus import UNet_3Plus


PATCH_H = 512
PATCH_W = 512
STRIDE_H = PATCH_H // 2
STRIDE_W = PATCH_W // 2
INFERENCE_BATCH_SIZE = 1


def _get_gaussian_weight_map(shape=(PATCH_H, PATCH_W)):
    center_y, center_x = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    y, x = np.ogrid[:shape[0], :shape[1]]
    sigma_y, sigma_x = shape[0] / 4, shape[1] / 4
    gaussian_map = np.exp(-(((y - center_y) ** 2 / (2. * sigma_y ** 2)) + ((x - center_x) ** 2 / (2. * sigma_x ** 2))))
    gaussian_map = (gaussian_map - gaussian_map.min()) / (gaussian_map.max() - gaussian_map.min())
    gaussian_map += 1e-6
    return gaussian_map.astype(np.float32)

gaussian_weight_map = _get_gaussian_weight_map((PATCH_H, PATCH_W))


def process_single_image(model, input_image_path, output_image_path):
    start_time = time.time()
    try:
        print(f"--- Starting Patch-Based Processing: {input_image_path} ---")
        print(f"Config: Patch Size=({PATCH_H},{PATCH_W}), Stride=({STRIDE_H},{STRIDE_W})")

        original_image = cv2.imread(input_image_path)
        if original_image is None:
            print(f"Error: Could not load image {input_image_path}")
            return False
        original_h, original_w, _ = original_image.shape
        print(f"Original image size: H={original_h}, W={original_w}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        img_normalized = np.float32(original_image) / 255.0

        n_rows = math.ceil((original_h - PATCH_H) / STRIDE_H) + 1 if original_h > PATCH_H else 1
        n_cols = math.ceil((original_w - PATCH_W) / STRIDE_W) + 1 if original_w > PATCH_W else 1
        padded_h = (n_rows - 1) * STRIDE_H + PATCH_H
        padded_w = (n_cols - 1) * STRIDE_W + PATCH_W
        pad_top = 0
        pad_bottom = padded_h - original_h
        pad_left = 0
        pad_right = padded_w - original_w

        print(f"Calculated Padded Size: H={padded_h}, W={padded_w}")
        padded_image = cv2.copyMakeBorder(img_normalized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)

        result_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
        result_counts = np.zeros((padded_h, padded_w), dtype=np.float32)

        total_patches = n_rows * n_cols
        processed_patches = 0
        patch_coords = []
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                y_start = r_idx * STRIDE_H
                y_end = y_start + PATCH_H
                x_start = c_idx * STRIDE_W
                x_end = x_start + PATCH_W
                patch_coords.append((y_start, y_end, x_start, x_end))

        weight_map_cpu = gaussian_weight_map
        weight_map_gpu = torch.from_numpy(weight_map_cpu).to(device)

        for y_start, y_end, x_start, x_end in patch_coords:
            processed_patches += 1
            if processed_patches % 20 == 0 or processed_patches == total_patches: 
                 print(f"Processing Patch {processed_patches}/{total_patches}...")

            image_patch = padded_image[y_start:y_end, x_start:x_end, :]
            patch_expanded = np.expand_dims(image_patch, 0)
            patch_input = torch.from_numpy(patch_expanded).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                logits = model(patch_input)
                probabilities = F.softmax(logits, dim=1)
                crack_probs = probabilities[:, 1, :, :]
                crack_probs = crack_probs.squeeze(0)

                weighted_patch_mask = crack_probs * weight_map_gpu

                result_sum[y_start:y_end, x_start:x_end] += weighted_patch_mask.cpu().numpy()
                result_counts[y_start:y_end, x_start:x_end] += weight_map_cpu

        print("All patches processed, calculating final mask...")

        result_counts[result_counts == 0] = 1e-6
        final_avg_prob = result_sum / result_counts

        final_mask = (final_avg_prob > 0.5).astype(np.uint8) * 255

        final_mask_cropped = final_mask[pad_top:pad_top + original_h, pad_left:pad_left + original_w]

        output_dir = os.path.dirname(output_image_path)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_image_path, final_mask_cropped)
        print(f"Processing result saved to: {output_image_path}")

        end_time = time.time()
        print(f"--- Patch-Based Processing successful, total time: {end_time - start_time:.2f} seconds ---")
        return True

    except Exception as e:
        print(f"ERROR during image processing: {e}")
        traceback.print_exc()
        print("--- Image processing failed ---")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

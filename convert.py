#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import shutil
from argparse import ArgumentParser

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action="store_true", help="Disable GPU for COLMAP")
parser.add_argument("--skip_matching", action="store_true", help="Skip feature extraction & matching")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to input folder containing 'input' images")
parser.add_argument("--camera", default="OPENCV", type=str, help="Camera model to use in COLMAP")
parser.add_argument("--colmap_executable", default="", type=str, help="Path to COLMAP binary")
parser.add_argument("--resize", action="store_true", help="Also generate downsampled versions of images")
parser.add_argument("--magick_executable", default="", type=str, help="Path to ImageMagick binary")
parser.add_argument("--mask_path", type=str, default=None, help="Optional path to folder with masks aligned to input images")
args = parser.parse_args()

colmap_command = f"\"{args.colmap_executable}\"" if args.colmap_executable else "colmap"
magick_command = f"\"{args.magick_executable}\"" if args.magick_executable else "magick"
use_gpu = 1 if not args.no_gpu else 0

# --------------------
# COLMAP pipeline
# --------------------
if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted/sparse"), exist_ok=True)

    # build mask argument if provided
    mask_arg = f"--ImageReader.mask_path {args.mask_path}" if args.mask_path else ""

    feat_extracton_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {args.camera} "
        f"--SiftExtraction.use_gpu {use_gpu} "
        f"{mask_arg}"
    )
    if os.system(feat_extracton_cmd) != 0:
        logging.error("Feature extraction failed.")
        exit(1)

    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    if os.system(feat_matching_cmd) != 0:
        logging.error("Feature matching failed.")
        exit(1)

    mapper_cmd = (
        f"{colmap_command} mapper "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--output_path {args.source_path}/distorted/sparse "
        f"--Mapper.ba_global_function_tolerance=0.000001"
    )
    if os.system(mapper_cmd) != 0:
        logging.error("Mapper failed.")
        exit(1)

# --------------------
# Undistortion
# --------------------
img_undist_cmd = (
    f"{colmap_command} image_undistorter "
    f"--image_path {args.source_path}/input "
    f"--input_path {args.source_path}/distorted/sparse/0 "
    f"--output_path {args.source_path} "
    f"--output_type COLMAP"
)
if os.system(img_undist_cmd) != 0:
    logging.error("Image undistortion failed.")
    exit(1)

# Move sparse files into sparse/0
sparse_dir = os.path.join(args.source_path, "sparse")
os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
for file in os.listdir(sparse_dir):
    if file == "0":
        continue
    shutil.move(os.path.join(sparse_dir, file), os.path.join(sparse_dir, "0", file))

# --------------------
# Handle masks (optional copy for reference)
# --------------------
if args.mask_path:
    masks_out = os.path.join(args.source_path, "masks")
    os.makedirs(masks_out, exist_ok=True)
    for file in os.listdir(args.mask_path):
        src = os.path.join(args.mask_path, file)
        dst = os.path.join(masks_out, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print(f"Copied masks from {args.mask_path} → {masks_out}")

# --------------------
# Resizing (optional)
# --------------------
if args.resize:
    print("Copying and resizing images...")
    for factor, folder in [(2, "images_2"), (4, "images_4"), (8, "images_8")]:
        out_dir = os.path.join(args.source_path, folder)
        os.makedirs(out_dir, exist_ok=True)
        for file in os.listdir(os.path.join(args.source_path, "images")):
            src = os.path.join(args.source_path, "images", file)
            dst = os.path.join(out_dir, file)
            shutil.copy2(src, dst)
            scale = int(100 / factor)
            resize_cmd = f"{magick_command} mogrify -resize {scale}% {dst}"
            if os.system(resize_cmd) != 0:
                logging.error(f"Resize {scale}% failed for {file}")
                exit(1)

print("Done.")

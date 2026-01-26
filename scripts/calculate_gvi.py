"""
Script to calculate Green View Index (GVI) for Al Karama SVI images using ZenSVI.
"""

import os
import glob
from zensvi.cv import Segmenter

# Input directory with downloaded SVI images
base_input_dir = "data/svi_images/mly_svi"

# Output directory for segmentation results
output_dir = "data/segmentation"
os.makedirs(output_dir, exist_ok=True)

print(f"Input base directory: {base_input_dir}")
print(f"Output directory: {output_dir}")

# Initialize the segmenter with Mapillary dataset (better for street-level imagery)
segmenter = Segmenter(
    dataset="mapillary",  # Options: "mapillary" or "cityscapes"
    task="semantic"       # Semantic segmentation to identify vegetation
)

# Find all batch directories
batch_dirs = sorted(glob.glob(os.path.join(base_input_dir, "batch_*")))
print(f"\nFound {len(batch_dirs)} batch directories")

# Process each batch
for i, batch_dir in enumerate(batch_dirs, 1):
    batch_name = os.path.basename(batch_dir)
    print(f"\n[{i}/{len(batch_dirs)}] Processing {batch_name}...")

    segmenter.segment(
        dir_input=batch_dir,
        dir_image_output=os.path.join(output_dir, "images", batch_name),
        dir_summary_output=os.path.join(output_dir, "summary", batch_name),
        save_image_options="segmented_image blend_image",
        save_format="csv json",
        csv_format="long"
    )

print(f"\nGreen View Index calculation complete!")
print(f"Summary results saved to: {output_dir}/summary/")
print(f"Segmented images saved to: {output_dir}/images/")

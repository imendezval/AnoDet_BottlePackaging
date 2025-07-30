"""
Image Augmentation with Brightness and Contrast Adjustments.

This script was specifically created to augument the instance where the
conveyor belt is empty - since only 40 instances where found in the  
>6000 image dataset, even though it is the most common instance by far.
It works for all other images too.

This script performs the following tasks:
1. Reads images from an input folder.
2. Generates multiple augmented versions of each image by randomly adjusting brightness and contrast.
3. Reapplies the mask to the augmented images, since black region is affected
4. Saves the augmented images into an output folder.

Functions:
- `adjust_brightness_contrast(image, alpha, beta)`: Adjusts brightness and contrast of an image.
"""
import os
import cv2
import argparse
import numpy as np

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Brightness/Contrast Augmentation with Masking")

        parser.add_argument("--input-folder", type=str, required=True, help="Path to folder with original images")
        parser.add_argument("--output-folder", type=str, required=True, help="Path to save augmented images")
        parser.add_argument("--mask-path", type=str, required=True, help="Path to mask image")
        parser.add_argument("--num-variants", type=int, default=4, help="Number of augmented versions per image")
        parser.add_argument("--alpha-range", type=float, nargs=2, default=[0.8, 1.2], help="Range for contrast adjustment (alpha)")
        parser.add_argument("--beta-range", type=int, nargs=2, default=[-40, 40], help="Range for brightness adjustment (beta)")

        self.args = parser.parse_args()

class ImageAugmentor:
    def __init__(self, config: Config):
        self.cfg = config.args
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        self.mask = cv2.imread(self.cfg.mask_path)
        if self.mask is None:
            raise FileNotFoundError(f"Mask not found at {self.cfg.mask_path}")

    def adjust_brightness_contrast(self, image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def augment_images(self):
        image_files = [f for f in os.listdir(self.cfg.input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for image_file in image_files:
            image_path = os.path.join(self.cfg.input_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading {image_file}")
                continue

            for i in range(self.cfg.num_variants):
                alpha = np.random.uniform(*self.cfg.alpha_range)
                beta = np.random.randint(*self.cfg.beta_range)

                aug_img = self.adjust_brightness_contrast(image, alpha, beta)
                aug_masked = cv2.bitwise_and(aug_img, self.mask)

                base_name = os.path.splitext(image_file)[0]
                out_name = f"aug_{i}_{base_name}.jpg"
                out_path = os.path.join(self.cfg.output_folder, out_name)

                cv2.imwrite(out_path, aug_masked)
                print(f"Saved: {out_path}")

def main():
    config = Config()
    augmenter = ImageAugmentor(config)
    augmenter.augment_images()

if __name__ == "__main__":
    main()
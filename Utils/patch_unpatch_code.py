import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image


def read_tif_file(image_file_path):
    image = Image.open(image_file_path)
    image_data = np.array(image)
    print(image_data.shape)
    rgb_image = image_data[:, :, :3]
    img_array = np.moveaxis(rgb_image, 0, 2)
    img_array = np.uint8(rgb_image)
    return img_array


def patch_(image_file_path,  patch_size, patch_files_dir_path):
    #os.makedirs(patch_files_dir_path, exist_ok=True)
    image_array = read_tif_file(image_file_path)
    height, width, _ = image_array.shape
    old_shape = image_array.shape
    print(f"Initial Shape : {old_shape}")

    x = math.ceil(int(height) / (int(patch_size[0])))
    y = math.ceil(int(width) / (int(patch_size[1])))

    left_border = (patch_size[1] * y) - width
    bottom_border = (patch_size[0] * x) - height
    print(f"Lett & Right Padding: {left_border, bottom_border}")

    if left_border != 0:
        image_array = cv2.copyMakeBorder(
            image_array, 0, 0, 0, left_border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    if bottom_border != 0:
        image_array = cv2.copyMakeBorder(
            image_array, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    new_shape = image_array.shape
    # padded_img = patch_image = Image.fromarray(image_array)
    # padded_img.save(os.path.join(padded_img_folder, f"padded.tif"))
    # print(f"After Padding Shape : {new_shape}")

    for i in range(0, height, patch_size[0]):
        for j in range(0, width, patch_size[1]):
            patch = image_array[i : i + patch_size[0], j : j + patch_size[1], :]
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(patch_files_dir_path, f"patch_{i}_{j}.tif"))

    return old_shape, new_shape


def unpatch_(image_file_path, old_shape, new_shape, patch_size, patch_files_dir_path):
    old_image_height, old_image_width, o_c = old_shape
    new_image_height, new_image_width, n_c = new_shape
    output_image = np.zeros((new_image_height, new_image_width, n_c), dtype=np.uint8)
    for i in range(0, new_image_height, patch_size[0]):
        for j in range(0, new_image_width, patch_size[1]):
            patch_file_path = os.path.join(patch_files_dir_path, f"patch_{i}_{j}.tif")
            if os.path.exists(patch_file_path):
                image = Image.open(patch_file_path)
                image_array = np.array(image)
                output_image[
                    i : i + patch_size[0], j : j + patch_size[1], :
                ] = image_array
    output_image = output_image[:old_image_height, :old_image_width, :]
    output_image = Image.fromarray(output_image)
    output_image.save(image_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file-path",
        type=str,
        default="rgb.tif",
        help="Input TiF Image File Path!",
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default="Output_Testing_Image.tif",
        help="Output TiF Image File Path!",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=640,
        help="Patch Size",
    )
    parser.add_argument(
        "--patch-images-dir",
        type=str,
        default="patch_files",
        help="Dir Path to save Patch Image",
    )
    args = parser.parse_args()
    input_image_file_path = args.input_file_path
    output_image_file_path = args.output_file_path
    patch_files_dir = args.patch_images_dir
    patch_size = (args.patch_size, args.patch_size)
    old_shape, new_shape = patch_(
        input_image_file_path,
        patch_size,
        patch_files_dir,
    )
    unpatch_(
        output_image_file_path,
        old_shape,
        new_shape,
        patch_size,
        patch_files_dir,
    )

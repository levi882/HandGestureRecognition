import os
from PIL import Image, ImageOps


def mirror_images_recursively(root_dir):
    """
    Recursively walks through directories starting from root_dir,
    mirrors images and saves them with '_inv' suffix.
    Skips images that already have a mirrored version.
    """
    # Common image extensions to process
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    # Count for reporting
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Starting to process images in {root_dir} and its subdirectories...")

    # Walk through all directories and files recursively
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get all filenames in the current directory
        all_files_in_dir = set(filenames)
        
        for filename in filenames:
            # Check if the file is an image based on extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Full path to original image
                image_path = os.path.join(dirpath, filename)

                # Parse filename and extension
                name, ext = os.path.splitext(filename)

                # Skip if already an inverted image
                if name.endswith('_inv'):
                    continue

                # Path for the mirrored image
                mirrored_filename = f"{name}_inv{ext}"
                mirrored_path = os.path.join(dirpath, mirrored_filename)

                # Skip if mirrored version already exists
                if mirrored_filename in all_files_in_dir:
                    skipped_count += 1
                    print(f"Skipped: {image_path} (mirrored version already exists)")
                    continue

                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Mirror the image horizontally
                        mirrored_img = ImageOps.mirror(img)

                        # Save the mirrored image
                        mirrored_img.save(mirrored_path)

                    processed_count += 1
                    print(f"Mirrored: {image_path} -> {mirrored_path}")
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {image_path}: {e}")

    print("\nProcessing complete!")
    print(f"Total images mirrored: {processed_count}")
    print(f"Images skipped (already mirrored): {skipped_count}")
    print(f"Errors encountered: {error_count}")


if __name__ == "__main__":
    # Set the root directory directly in the code
    root = "./data/custom"  # Change this to your actual directory path

    # Check if the provided path exists
    if not os.path.exists(root):
        print(f"Error: The path {root} does not exist.")
        exit(1)

    # Process images
    mirror_images_recursively(root)

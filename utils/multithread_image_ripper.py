import os
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def download_images(image_urls, folder_loc, max_workers=10, convert_to_jpg=False):
    """
    Download a list of images from given URLs and save them to a specified folder using multithreading.
    Optionally convert .webp images to .jpg format.

    Parameters:
    - image_urls (list): List of image URLs to download.
    - folder_loc (str): Directory path to save the downloaded images.
    - max_workers (int): Number of threads to use for downloading. Default is 10.
    - convert_to_jpg (bool): If True, convert .webp images to .jpg format. Default is False. (Figured out YOLO needs this)
    """
    # Ensure the folder exists
    #TODO: make folder_loc robust for names with slashes e.g. Fiat X1/9
    os.makedirs(folder_loc, exist_ok=True)

    def download_and_convert_image(url):
        """
        Helper function to download a single image from the given URL and optionally convert .webp to .jpg.
        """
        try:
            # Get the image data
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Extract the filename from the URL
            original_filename = url.split('_')[-1].replace('/', '-')
            filepath = os.path.join(folder_loc, original_filename)

            # Save the image
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            # Convert .webp to .jpg if needed
            if convert_to_jpg and filepath.lower().endswith('.webp'):
                jpg_path = filepath.rsplit('.', 1)[0] + '.jpg'
                with Image.open(filepath) as img:
                    img.convert("RGB").save(jpg_path, "JPEG")
                os.remove(filepath)  # Remove the original .webp file
                print(f"Converted: {filepath} -> {jpg_path}")
            else:
                print(f"Downloaded: {filepath}")

        except Exception as e:
            print(f"Failed to process {url}: {e}")

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_and_convert_image, image_urls)

from PIL import Image
import os

def find_extreme_resolutions(folder_path):
    """
    Read all images in a dataset and identify the largest and smallest resolutions
    based on pixel count (width × height) across all images.
    
    Args:
        folder_path (str): Path to the folder containing images.
        
    Returns:
        dict: A dictionary summarizing the largest and smallest resolutions, including
              filenames, resolutions, and pixel counts. In case of errors, returns an error message.
    """
    try:
        # Supported image formats
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_resolutions = []
        
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            return f"Error: The folder '{folder_path}' does not exist."
        
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Read the image
                    with Image.open(file_path) as img:
                        # Get image resolution (width, height)
                        width, height = img.size
                        pixel_count = width * height
                        
                        # Store resolution and pixel count
                        image_resolutions.append({
                            'filename': filename,
                            'resolution': (width, height),
                            'pixel_count': pixel_count
                        })
                except Exception as e:
                    print(f"Warning: Could not read '{filename}': {str(e)}")
        
        if not image_resolutions:
            return f"Error: No valid images found in the folder '{folder_path}'."
        
        # Find the largest and smallest resolutions based on pixel count
        largest = max(image_resolutions, key=lambda x: x['pixel_count'])
        smallest = min(image_resolutions, key=lambda x: x['pixel_count'])
        
        # Summarize results
        summary = {
            'largest_resolution': {
                'filename': largest['filename'],
                'resolution': largest['resolution'],
                'pixel_count': largest['pixel_count']
            },
            'smallest_resolution': {
                'filename': smallest['filename'],
                'resolution': smallest['resolution'],
                'pixel_count': smallest['pixel_count']
            }
        }
        
        return summary
    except Exception as e:
        return f"Error processing folder: {str(e)}"

# Example usage
folder_path = "dataset/peta/images"  # Replace with the path to your dataset folder
results = find_extreme_resolutions(folder_path)

# Print the results
if isinstance(results, dict):
    print("Largest Resolution:")
    print(f"  Filename: {results['largest_resolution']['filename']}")
    print(f"  Resolution: {results['largest_resolution']['resolution'][0]}x"
          f"{results['largest_resolution']['resolution'][1]} pixels")
    print(f"  Pixel Count: {results['largest_resolution']['pixel_count']} pixels")
    print("\nSmallest Resolution:")
    print(f"  Filename: {results['smallest_resolution']['filename']}")
    print(f"  Resolution: {results['smallest_resolution']['resolution'][0]}x"
          f"{results['smallest_resolution']['resolution'][1]} pixels")
    print(f"  Pixel Count: {results['smallest_resolution']['pixel_count']} pixels")
elif isinstance(results, str):
    print(results)
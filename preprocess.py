from PIL import Image
from streetview import search_panoramas
from streetview import get_panorama
import random, math

def generate_spread_points(latitude, longitude, radius=2000, num_points=10):
    random_points = []
    earth_radius = 6371000  # Earth's radius in meters

    for _ in range(num_points):
        # Randomly choose a distance from the center, ensuring points are spread out
        random_distance = radius * math.sqrt(random.uniform(0, 1))

        # Random angle in radians
        random_angle = random.uniform(0, 2 * math.pi)

        # Latitude offset
        delta_lat = random_distance * math.cos(random_angle) / earth_radius
        new_lat = latitude + delta_lat * (180 / math.pi)

        # Longitude offset
        delta_long = random_distance * math.sin(random_angle) / (earth_radius * math.cos(math.radians(latitude)))
        new_long = longitude + delta_long * (180 / math.pi)

        random_points.append((new_lat, new_long))

    return random_points
from PIL import Image

from PIL import Image

def un_panoramize(pil_image, target_aspect_ratio=(16, 9), crop_strength=1, zoom_factor=2, final_size=(1920, 1080), downscale_factor=0.25):
    # Downscale the image for lower memory usage
    small_width = int(pil_image.width * downscale_factor)
    small_height = int(pil_image.height * downscale_factor)
    pil_image = pil_image.resize((small_width, small_height), Image.LANCZOS)
    
    # Convert PIL image to RGB mode if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Get dimensions after downscaling
    width, height = pil_image.size

    # Calculate the target width based on the target aspect ratio and crop strength
    target_width = int(height * (target_aspect_ratio[0] / target_aspect_ratio[1]) * crop_strength)
    
    # Ensure target width does not exceed the image width
    target_width = min(target_width, width)

    # Adjust the crop starting point to focus on the upper part of the image
    start_x = (width - target_width) // 2
    start_y = int(height * 0.1)  # Shifts crop start a bit downward to keep more of the top

    # Crop the image to achieve the de-panoramized effect with upper focus
    cropped_image = pil_image.crop((start_x, 0, start_x + target_width, start_y + height))

    # Apply zoom to the cropped image
    zoomed_width = int(cropped_image.width * zoom_factor)
    zoomed_height = int(cropped_image.height * zoom_factor)
    zoomed_image = cropped_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)

    # Calculate the center crop after zooming to return to the original target aspect ratio
    final_start_x = (zoomed_image.width - target_width) // 2
    final_start_y = 0  # Keep the top border intact
    final_image = zoomed_image.crop((final_start_x, final_start_y, final_start_x + target_width, final_start_y + height))

    # Downscale to final size
    final_image = final_image.resize(final_size, Image.LANCZOS)

    return final_image



def get_image_from_cords(latt, long, path):
    try:
        panorama_data = search_panoramas(latt, long)
        
        panorama_id = panorama_data[0].pano_id

        image = get_panorama(pano_id=panorama_id)

        image = un_panoramize(image)

        image.save(path, format="JPEG")
    
    except Exception as e:
        print(e)
        return None, False



points = generate_spread_points(2.294694, 48.858093)
i = 0
for point in points:

    i = i + 1

    latt = point[1]
    long = point[0]

    path = f"/home/metricspace/HackatonHamburg/testimg/imgs_{i}.jpg"

    image = get_image_from_cords(latt, long, path)
        

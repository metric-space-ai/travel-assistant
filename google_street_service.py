import os


# Install dependencies
dependencies = [
    "pip install -q regex",
    "pip install -q flask",
    "pip install -q transformers",
    "pip install -q torch",
    "pip install -q Pillow",
    "pip install -q marker-pdf",
    "pip install -q langchain-community",
    "pip install -q streetview",
    "pip install -q werkzeug",
    "pip install matplotlib",
    "pip install openai"
]

for command in dependencies:
    os.system(command)
    
    
import json
import os
import re
from streetview import search_panoramas
from streetview import get_panorama
import base64
from streetview import get_panorama_async
import matplotlib.pyplot as plt
from openai import OpenAI
import random
import math
import json
from PIL import Image
import subprocess
import threading
from flask import Flask, jsonify, request
import base64
import requests
from io import BytesIO


config_str = '''{
"device_map": {
    "cuda:0": "15GiB",
    "cuda:1": "15GiB",
    "cuda:2": "15GiB",
    "cuda:3": "15GiB"
},
"required_python_version": "cp311",
"models": [
    {
    "name": "ollama:llama3.1:8b"
    }
],
"functions": [
    {
    "name": "get_scores",
    "description": "returns scores about streetview images",
    "parameters": {
        "type": "object",
        "properties": {
        "long": {
            "type": "string",
            "description": "Longitude of location"
        },
        "latt": {
            "type": "string",
            "description": "Latitude of location"
        }
        }
    },
    "input_type": "application/pdf",
    "return_type": "application/json"
    }
]
}

'''

with open("config.json") as config_file:

    config_secret = json.load(config_file)



os.environ["SUPERPROXY_ISP_USER"] = config_secret["SUPERPROXY_ISP_USER"]

os.environ["SUPERPROXY_ISP_PASSWORD"] = config_secret["SUPERPROXY_ISP_PASSWORD"]

os.environ["SUPERPROXY_SERP_USER"] = config_secret["SUPERPROXY_SERP_USER"]

os.environ["SUPERPROXY_SERP_PASSWORD"] = config_secret["SUPERPROXY_SERP_PASSWORD"]

os.environ["OPENAI_API_KEY"] = config_secret["openai_api_key"]




config = json.loads(config_str)


openai_key = os.getenv('OPENAI_API_KEY')
token = os.getenv('OCTOPUS_TOKEN', '')

client = OpenAI()


app = Flask(__name__)



def create_scores(base64_image):


    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": """Critically evaluate the image on a scale from 1 to 5, focusing on factors that enhance or detract from its charm. Lower scores (1-2) indicate the presence of visual distractions like cars, clutter, or dull scenery, while higher scores (4-5) reflect vibrant colors, lush greenery, and appealing backdrops. Provide a score and a brief description in JSON format:
                        1.	Visual_appeal: Rate based on colors, scenery, and overall charm.
                        2.	Uniqueness_of_motif: Assess how memorable and distinctive the location is.
                        3.	Compositional_possibilities: Consider the potential for creative framing and clear views.
                        4.	Lighting_conditions: Evaluate the quality of natural light, shadows, and highlights.
                        5.	Interaction_potential: Rate how well the setting supports social activities or relaxation.
                        6.	Cultural_aesthetic_significance: Check for any artistic, historical, or cultural value.
                        7.	Adaptability_for_themes: Assess the location's versatility for various themes (e.g., weddings, travel).
                        8.	Self_staging_potential: Rate suitability for personal photoshoots or social media content.
                        {
                    “Visual_appeal”: {“score”: ..., “description”: “...“},
                    “Uniqueness_of_motif”: {“score”: ..., “description”: “...“},
                    “Compositional_possibilities”: {“score”: ..., “description”: “...“},
                    “Lighting_conditions”: {“score”: ..., “description”: “...“},
                    “Interaction_potential”: {“score”: ..., “description”: “...“},
                    “Cultural_aesthetic_significance”: {“score”: ..., “description”: “...“},
                    “Adaptability_for_themes”: {“score”: ..., “description”: “...“},
                    “Self_staging_potential”: {“score”: ..., “description”: “...“}
                    }""",
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
            },
            },
        ],
        }
    ],
    )
    
    text = response.choices[0].message.content
    print(text)
    try:
        return json.loads(text)
    except:
        try:
            cleaned_json_string = text.strip("```json\n")
            return json.loads(cleaned_json_string)
        except:
            return None
        
        

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



def load_image_from_url(url):
    response = requests.get(url)  # Fetch the image data from the URL
    response.raise_for_status()    # Ensure the request was successful
    img = Image.open(BytesIO(response.content))  # Open the image from the bytes data
    return img

def get_image_from_cords(latt, long):
    try:
        panorama_data = search_panoramas(latt, long)
        
        panorama_id = panorama_data[0].pano_id

        image = get_panorama(pano_id=panorama_id)

        # Process the image
        image = un_panoramize(image)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=70)

        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64, True
    
    except Exception as e:
        print(e)
        return None, False


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
                                                        

def generate_spread_points(latitude, longitude, radius=500, num_points=1):
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


def generate_and_store_evaluated_images(latitude, longitude):#, radius=500, num_points=1)

    print('getting image')
    image, status = get_image_from_cords(latitude, longitude)
    if status:
        scores = create_scores(image)
        json_data = {"long": longitude, "latt": latitude, "scores": scores}

    else:
        return "Wrong coordinates", False   

        
    return json_data, True


def save_on_server(result, cords):
    url = 'https://api.likora.octopus-ai.app/api/v1/kvs'

    # Convert result and cords to strings since kv_key and kv_value need to be strings
    kv_key = f"{cords[0]},{cords[1]}"
    kv_value = json.dumps(result)  # Convert dictionary to JSON string

    payload = {
        "access_type": "Company",
        "expires_at": "2024-11-13T15:53:30.536Z",  # Ensure date is in correct format
        "kv_key": kv_key,
        "kv_value": kv_value
    }

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'X-Auth-Token': token  # Ensure this is a valid token
    }

    response = requests.post(url, headers=headers, json=payload)

    # Print response details
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")


@app.route('/get_scores', methods=['POST'])
def upload_pdf():
    
    data = request.json  # URL
    long = data.get("long")
    latt = data.get("latt")
    cords = (latt, long)
    try:
        result, should_be_added = generate_and_store_evaluated_images(float(latt), float(long))
        print(result)
        if should_be_added:
            save_on_server(result, cords)
        return jsonify({"response": json.dumps(result)}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False

def run_flask_in_thread():
    flask_thread = threading.Thread(target=start_app)
    flask_thread.start()



@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }
    return jsonify(response), 201


import requests

def test_get_scores():
    print("STARTED")
    # Define the URL for the local Flask server
    url = "http://127.0.0.1:5000/get_scores"
    
    # Sample latitude and longitude data for the request
    sample_data = {
        "latt": "48.858093",
        "long": "2.294694"
    }
    
    try:
        # Send a POST request to the get_scores endpoint
        response = requests.post(url, json=sample_data)
        print(response.status_code)
        #print(response.json())
    
    except Exception as e:
        print(f"Test encountered an error: {e}")

# Run the test function
# Start the Flask app in a separate thread
import time
run_flask_in_thread()
print('ok')
time.sleep(3)  # Adjust the delay if needed
test_get_scores()

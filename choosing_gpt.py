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
    "pip install selenium selenium-wire bs4 html2text nltk",
    "pip install python-dotenv",
    "pip install openai",
    "pip install pymupdf"
]

for command in dependencies:
    os.system(command)
    


import heapq
import json
import threading
import requests
from flask import Flask, request, jsonify
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from openai import OpenAI



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
    "name": "pick_best_place",
    "description": "picks a best place based on locations",
    "parameters": {
        "type": "object",
        "properties": {
        "list_of_chords": {
            "type": "string",
            "description": "coordinates"
        },
        "user_pref": {
            "type": "string",
            "description": "user_preferences"
        },        
        "k_numbers_of_locations": {
            "type": "string",
            "description": "number of returned places"
        }
        }
    },
    "input_type": "application/pdf",
    "return_type": "application/json"
    }
]
}
'''

config = json.loads(config_str)



app = Flask(__name__)

# Load the sentence-transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set your OpenAI API key and token

token =  os.getenv('OCTOPUS_TOKEN')
openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def find_best_location_match(user_input, json_list, k):
    # Encode the user input
    user_embedding = model.encode([user_input], convert_to_tensor=True)
    
    # Initialize a list to store all matches with their scores
    matches = []
    
    for item in json_list:
        for coordinates, details in item.items():
            # Parse the details string as JSON
            details_data = json.loads(details)
            
            # Extract descriptions from "scores" field
            descriptions = [
                entry["description"] 
                for entry in details_data["scores"].values()
            ]
            
            # Join descriptions into a single text block
            combined_description = " ".join(descriptions)
            
            # Encode the combined description text
            description_embedding = model.encode([combined_description], convert_to_tensor=True)
            
            # Compute the cosine similarity
            score = F.cosine_similarity(user_embedding, description_embedding).item()
            
            # Add the coordinates and score to the matches list
            matches.append((score, coordinates))
    
    # Get the top k matches based on the score
    best_matches = heapq.nlargest(k, matches, key=lambda x: x[0])
    
    # Return only the coordinates of the top k matches
    return [coordinates for _, coordinates in best_matches]

def fetch_key_value_pairs():
    url = 'https://api.likora.octopus-ai.app/api/v1/kvs'

    headers = {
        'accept': 'application/json',
        'X-Auth-Token': token  # Use a valid token
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        
        # Extract key-value pairs
        list_of_pairs = []
        key_value_pairs = [{item['kv_key']: item['kv_value']} for item in data]
        for pair in key_value_pairs:
            list_of_pairs.append(pair)

        return list_of_pairs
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return []

def inference_get_scores(chords):
    print("INFERENCING")
    url = 'https://api.likora.octopus-ai.app/api/v1/ai-functions/direct-call'

    headers = {
        'accept': 'application/json',
        'X-Auth-Token': token,
    }

    latt = chords[0]
    long = chords[1]

    payload = {
        "name": "get_scores",
        "parameters": {
            "long": long,
            "latt": latt
        }
    }

    response = requests.post(url, json=payload, headers=headers)  
    data = response.json()
    print("OUTPUT")
    if data == {'Mixed': [{'Text': {'response': '"Wrong coordinates"'}}]}:
        return None
    print(data)
    return eval(data['Mixed'][0]['Text']['response'])['scores']

def complete_data(data_list, coordinates):
    import json
    
    # Create a set of user coordinate strings for quick lookup
    user_coords_set = {f"{lat},{long}" for lat, long in coordinates}
    
    # Create a dictionary of existing data with coordinate strings as keys, only if they are in user input
    data_dict = {key: json.loads(value) for item in data_list for key, value in item.items() if key in user_coords_set}
    
    # Process each coordinate in the input list and add missing data
    for lat, long in coordinates:
        coord_str = f"{lat},{long}"
        if coord_str not in data_dict:
            # Generate data only for missing coordinates
            output = inference_get_scores((lat, long))
            if output is not None:
                data_dict[coord_str] = output
            else:
                print(f"No data generated for {coord_str}\n")
    
    # Convert data_dict back to the list format, including only user-specified coordinates
    updated_data_list = [{coord_str: json.dumps(data_dict[coord_str])} for coord_str in user_coords_set if coord_str in data_dict]
    
    # Print updated_data_list for verification
    print("Updated data_list:")
    for item in updated_data_list:
        print(item)
    
    return updated_data_list



def choose_the_best(list_of_chords, user_pref, k):
    existing_data = fetch_key_value_pairs()
    completed_data = complete_data(existing_data, list_of_chords)
    return find_best_location_match(user_pref, completed_data, k)

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }
    return jsonify(response), 201



@app.route('/choose_best', methods=['POST'])
def upload_pdf():
    
    data = request.get_json()
    list_of_chords = data.get('list_of_chords', [])
    user_pref = data.get('user_pref', '')
    k_numbers_of_locations = data.get('k_numbers_of_locations', 1)

    try:
        # Convert list of lists to list of tuples
        list_of_chords = [tuple(coord) for coord in list_of_chords]

        result = choose_the_best(list_of_chords, user_pref, int(k_numbers_of_locations))

        return jsonify({"response": json.dumps(result)}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False

def run_flask_in_thread():
    flask_thread = threading.Thread(target=start_app)
    flask_thread.start()





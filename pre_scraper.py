import requests

from requests.adapters import HTTPAdapter



token = "ce160836-0c6d-4176-b803-4ab9785d19ce"

url = 'https://api.likora.octopus-ai.app/api/v1/ai-functions/direct-call'




def generate_score_sevice(lat, long):

    headers = {
        'accept': 'application/json',
        'X-Auth-Token': token,
    }
    payload = {
        "name": "get_scores",
        "parameters": {
            "latt": lat,
            "long": long
        }
    }

    response = requests.post(url, json=payload, headers=headers)  
    print(f'Generate a Scoring and saved on Server for cords {lat}, {long}')


def send_recommendation_request(event_type, latt, long, prompt):
    # Replace with the actual URL where your Flask app is running
    url = 'http://localhost:5000/recommendation'

    # Data to be sent in the POST request
    data = {
        "event": event_type,
        "latitude": latt,
        "longitude": long,
        "prompt": prompt
    }

    headers = {'Content-Type': 'application/json'}

    try:
        # Send POST request
        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None




travel_plan_events = [
    "Museum visits",
    "Restaurant dining",
    "Park visits",
    "City tours",
    "Historical site visits",
    "Shopping excursions",
    "Beach outings",
    "Hiking trips",
    "Amusement parks",
    "Food tastings",
    "Nightlife experiences",
    "Boat tours",
    "Art galleries",
    "Street markets",
    "Wildlife watching",
    "Spa sessions",
    "Photography tours",
    "Wine tasting",
    "Botanical gardens",
    "Cooking classes",
    "Adventure sports",
    "Scenic viewpoints",
    "Local workshops",
    "Temple visits",
    "Craft markets",
    "Farm tours"
]


import json 

with open('/home/metricspace/HackatonHamburg/places.json') as f:
    data_dict = json.load(f)






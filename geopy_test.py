from geopy.geocoders import Nominatim# Initialize the geocoder

geolocator = Nominatim(user_agent="geoapy")# Enter your address
address = "Hamburg Hans"# Get the location
location = geolocator.geocode(address)# Print the results


if location:
    print(f"Address: {location.address}")
    print(f"Latitude: {location.latitude}")
    print(f"Longitude: {location.longitude}")
else:
    print("Location not found")




import math

def haversine_distance(first, second):
    
    address = "Hamburg Hans"# Get the location
    location = geolocator.geocode(address)#
    lat1, lon1, lat2, lon2   # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    
    return distance



haversine_distance(lat1, lon1, lat2, lon2)
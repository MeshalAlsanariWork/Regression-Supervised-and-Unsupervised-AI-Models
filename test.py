import requests

url = 'https://lr-mode.onrender.com/predict'

data = {
    "appearance": 56,
    "minutes_played": 32432,
    "award": 5
}

response = requests.post(url, json=data)

print(response.json())
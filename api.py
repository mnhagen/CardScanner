import requests

url = "https://api.scryfall.com/cards/named"
params = {
    "fuzzy": "Black Lotus"
}

response = requests.get(url, params=params)

print(response.status_code)
print(response.json())
import requests

url = "http://34.72.217.0:8000/generate"  # Make sure the URL is correct
payload = {
    "prompt": "If i kill my best friend by mistakenly what will happen."
}

response = requests.post(url, json=payload)

# Print the status code and raw text of the response
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

# Now try to parse the JSON if it's valid
try:
    print(response.json())
except ValueError as e:
    print(f"Error decoding JSON: {e}")
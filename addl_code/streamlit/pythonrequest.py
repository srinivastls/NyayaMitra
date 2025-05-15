import requests

# Define the URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/get-judgments-summary/"

# Define the data to be sent in the request
data = {
    "judgment": "Some judgment text"  # Replace this with your actual judgment text
}

# Send the POST request
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print("Response from server:")
    print(response.text)  # Print the response text
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print the error message if any

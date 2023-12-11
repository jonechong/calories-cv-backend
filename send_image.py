import requests
from pathlib import Path


def send_image_to_api(image_path):
    url = "http://192.168.1.38:8000/predict"
    files = {"file": open(image_path, "rb")}

    response = requests.post(url, files=files)

    # Print the status code and response text for debugging
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Invalid JSON response", "response_text": response.text}


# Replace this with the path to your image file
image_path = Path("./data/test/CSF.jpg")

# Send the image to the API and get the prediction
result = send_image_to_api(image_path)
print("Prediction:", result)

import requests

url = "http://127.0.0.1:5003/predict"
files = {"file": open("../test/normal/image_280.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
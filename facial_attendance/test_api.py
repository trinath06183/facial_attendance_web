import requests
import cv2
import base64
import json
import time
import numpy as np
import time

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
_, buffer = cv2.imencode('.jpg', img)
b64_str = base64.b64encode(buffer).decode('utf-8')
data_url = f"data:image/jpeg;base64,{b64_str}"

try:
    response = requests.post("http://127.0.0.1:8000/process-client-frame/", json={"image": data_url})
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except:
        print("Response Text:", response.text)
except Exception as e:
    print("Request failed:", e)

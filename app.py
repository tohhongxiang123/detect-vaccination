from flask import Flask, request
from flask_cors import CORS
from predict import check_if_vaccinated_and_at_correct_location
import cv2
import numpy as np

app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# curl -X POST http://127.0.0.1:5000/predict/:location --form "image=@2008.jpg"
@app.route("/predict/<location>", methods=["POST"])
def a(location): 
    if 'image' not in request.files:
        return "No file was uploaded", 400 # Bad request

    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('received/a.jpg', img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = check_if_vaccinated_and_at_correct_location(img, location)
    print(result, location)
    return str(result)

if __name__ == "__main__":
    app.run(debug=True)
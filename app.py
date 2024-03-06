from flask import Flask, jsonify, request
import numpy as np
from numpy import array
from torch import load, device, no_grad, Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, Linear, Dropout
from torch.nn.functional import softmax, relu, max_pool2d
from PIL import Image
from io import BytesIO
import base64
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# List of labels for the classes
labels = [
    'airplane', 'bicycle', 'book', 'bowtie', 'bucket',
    'butterfly', 'cake', 'car', 'cat', 'cell phone',
    'clock', 'crown', 'eye', 'face', 'fish', 'house',
    'ice cream', 'light bulb', 'pizza', 'river', 'star',
    'sun', 't-shirt', 'tree', 'windmill'
]

# DoodleNet model definition
class DoodleNet(Module):
    def __init__(self, num_classes=25):
        super(DoodleNet, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2d(128)
        self.fc1 = Linear(128 * 4 * 4, 512)
        self.fc2 = Linear(512, num_classes)
        self.dropout = Dropout(0.5)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(self.bn2(self.conv2(x)))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(self.bn3(self.conv3(x)))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(self.bn4(self.conv4(x)))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 4 * 4)
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = DoodleNet()
model.load_state_dict(load('doodlewarsV2.pth', map_location=device('cpu')))
model.eval().cpu()

@app.route('/')
def home():
    return "<h1> Working </h1>"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data and object index from the request
    image_data = request.json['img']
    object_idx = int(request.json['object_idx'])

    # Remove the data URI prefix from the image data
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    # Convert the base64 string to bytes and then to PIL Image
    image_bytes = base64.b64decode(image_data)
    image_pil = Image.open(BytesIO(image_bytes))

    # Convert the image to grayscale and resize it
    image_pil = image_pil.convert('L')
    new_size = (64, 64)  # Set the new size (e.g., 64x64)
    resized_image_pil = image_pil.resize(new_size, Image.BICUBIC)

    # Convert the image to a tensor
    tensor = Tensor(array(resized_image_pil))

    # Check if the tensor is empty (all zeros)
    if tensor.max() == 0:
        return jsonify({'score': 0})

    # Normalize the tensor
    tensor = tensor / tensor.max()

    # Perform the prediction
    with no_grad():
        logits = model(tensor.unsqueeze(0).unsqueeze(0))

    # Apply temperature scaling to the logits
    temperature = 1
    probs = softmax(logits / temperature, dim=1)

    # Scale the output probability to the range of 0 to 10
    scaled_output = probs[0][object_idx] * 9 + 1

    # Return the scaled prediction as JSON
    return jsonify({'score': float(scaled_output)})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask,jsonify,request
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import base64
import re

app = Flask(__name__)

labels = ['airplane',
 'bicycle',
 'book',
 'bowtie',
 'bucket',
 'butterfly',
 'cake',
 'car',
 'cat',
 'cell phone',
 'clock',
 'crown',
 'eye',
 'face',
 'fish',
 'house',
 'ice cream',
 'light bulb',
 'pizza',
 'river',
 'star',
 'sun',
 't-shirt',
 'tree',
 'windmill']

class DoodleNet(nn.Module):
    def __init__(self, num_classes=25):
        super(DoodleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DoodleNet()
model.load_state_dict(torch.load('doodlewarsV2.pth', map_location=torch.device('cpu')))
model.eval().cpu()


@app.route('/')
def home():
    return "<h1> Working </h1>"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.json['img']
    object_idx = int(request.json['object_idx'])
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    # Convert the base64 string to bytes and then to PIL Image
    image_bytes = base64.b64decode(image_data)
    image_pil = Image.open(BytesIO(image_bytes))
    # Convert the image to grayscale
    image_pil = image_pil.convert('L')
    # Resize the image using PIL
    new_size = (64, 64)  # Set the new size (e.g., 64x64)
    resized_image_pil = image_pil.resize(new_size, Image.BICUBIC)

    # Convert the image to a tensor
    tensor = torch.Tensor(np.array(resized_image_pil)) / 255.0
    # Perform the prediction
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).unsqueeze(0))
    # Scale the output to the range of 0 to 1
    scaled_output = torch.sigmoid(output[0][object_idx]) * 9 + 1
    print(output.argmax(dim=1))
    # Return the scaled prediction as JSON
    return jsonify({'score': float(scaled_output)})


if __name__ == '__main__':

    app.run(debug=True)

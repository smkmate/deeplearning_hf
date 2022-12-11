import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class GenderModel():
    def __init__(self):
        # Loading pytorch model from PTH file on CPU
        self.model = torch.load("resources/ResnetModel_gender.pth", map_location=torch.device('cpu'))

        # Creating the label encodeing
        self.race = ['Female', 'Male']

        # Image transformator
        self.transform = transforms.Compose([
            transforms.Resize(200),
            transforms.ToTensor(),
        ])

    def predict(self, img):
        # Transforming the detected image for predition
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)

        # Getting the prediction
        pred = self.model(img)
        pred = pred.detach().numpy()
        pred = np.round(pred)
        pred = int(pred)
        pred = self.race[pred]
        return pred

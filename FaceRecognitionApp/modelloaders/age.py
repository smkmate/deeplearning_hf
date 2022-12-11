import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class AgeModel():
    def __init__(self):
        # Loading pytorch model from PTH file on CPU
        self.model = torch.load("resources/ResnetModel_age.pth", map_location=torch.device('cpu'))

        # Creating the label encodeing
        self.age = ['0-2','10-19','20-29','3-9','30-39','40-49','50-59','60-69','70+']

        # Image transformator
        self.transform = transforms.Compose([
            transforms.Resize(200),
            transforms.ToTensor(),
        ])

    def predict(self, img):
        # Transforming the image
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)

        # Predicting the class
        pred = self.model(img)
        pred = pred.detach().numpy()
        pred = np.argmax(pred, axis=1)
        pred = int(pred)
        pred = self.age[pred]
        return pred

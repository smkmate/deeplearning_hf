import torch.nn as nn
import torch
from models.ClassificationBase import ImageClassificationBase


class FaceClassificationModel(ImageClassificationBase):
    def __init__(self, num_classes, h, base_model):
        super().__init__()
        # Loading the base model
        if base_model == "ResNet":
            self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        else:
            self.network = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

        # Setting the activation function based on the number of classes
        if (num_classes == 1):
           self.activation = nn.Sigmoid()
        else:
          self.activation = nn.Softmax(1)

        # Adding the additional layers for transfer learning
        if base_model == "ResNet":
          num_ftrs = self.network.fc.in_features        # number of output features of ResNet
          self.network.fc = nn.Sequential(
              nn.Linear(num_ftrs, h),
              nn.ReLU(inplace=True),
              nn.Linear(h,num_classes),  # for the n output
              self.activation
          )
        else:
           num_ftrs = 25088                             # number of output fearures of VGG
           self.network.classifier = nn.Sequential(
              nn.Linear(num_ftrs, 4096),
              nn.ReLU(inplace=True),
              nn.Dropout(p=0.5),
              nn.Linear(4096, 4096),
              nn.ReLU(inplace=True),
              nn.Dropout(p=0.5),
              nn.Linear(4096, num_classes),
              self.activation
         )
    def forward(self, xb):
        return self.network(xb)

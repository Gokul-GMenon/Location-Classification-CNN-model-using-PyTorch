# import libraries
import os
import torch
from torchvision.transforms import transforms
import PIL.Image as Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('MODEL\Model', map_location=torch.device(device))

transformer = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.RandomHorizontalFlip(),
    
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

path = 'Please enter the path of the JPEG image to test'

image = Image.open(path)

input_img = transformer(image)
output = model(input_img.unsqueeze(0))

classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

_, pred = torch.max(output, dim = 1)

prediction = classes[pred.item()]

print('Prediction - ', prediction)
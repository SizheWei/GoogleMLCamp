import torch.nn as nn
import torch
import cv2
import os
import PIL.Image as Image
from net.mattingnet import UNet
import numpy as np

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda')

model = UNet(3, 1).to(device)
PATH = './mattingmodel/weights_70.pth'
#model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.load_state_dict(torch.load(PATH))
print("load model")

x_path = 'C:/Users/76789/Desktop/test/001.png'
x = cv2.imread(x_path)
x = x.transpose((2, 0, 1))
x = torch.from_numpy(x)
x = x.unsqueeze(0)
x = x.to(device, dtype=torch.float)
out = model(x)
out = (out > 0.65) * 1
out=out.cpu()
x=x.cpu()
print("done")
my_result = out[0][0]


tmp = x[0].numpy().astype(np.uint8)
ori_image = cv2.merge([tmp[2],tmp[1],tmp[0]])

my_alpha = ((my_result.numpy())*255).astype(np.uint8)
final = cv2.merge([tmp[2],tmp[1],tmp[0],my_alpha])

final = cv2.cvtColor(final,cv2.COLOR_BGRA2RGB)
cv2.imwrite('C:/Users/76789/Desktop/001matting.png', final)




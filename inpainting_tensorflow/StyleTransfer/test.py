from neural_style import stylize

import torch

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


style_n=10
model='./checkpoints/epoch_5_Wed_Jan_15_160613_2020_100000_10000000000.model'
cuda=1
device = torch.device("cuda" if cuda else "cpu")

with torch.no_grad():
    style_model = TransformerNet(style_num=style_n)
    state_dict = torch.load(model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    savepath='./output'

    for i in range(style_n):
        style_i=i
        print(i)
        for j in range(1,2):
            output_image=str(i)+str(j)
            content_img='C:/Users/76789/Desktop/posterimg/001.png'
            stylize(content_img,style_n,style_model,style_i,output_image,savepath,cuda)



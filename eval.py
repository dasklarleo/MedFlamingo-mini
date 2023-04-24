from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

import json
import matplotlib.pyplot as plt
import torch
from flamingo_mini.utils import load_url
from PIL import Image
from IPython.display import Image as Image_display
from transformers.configuration_utils import PretrainedConfig
from IPython import embed

model = FlamingoModel.from_pretrained("/public/bme/home/liuyx7/project/MedFlamingo-mini/training/flamingo-coco/checkpoint-1875")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model.to(device)
model.eval()

processor = FlamingoProcessor(model.config)


def eval():
    image=Image.open("/public/bme/home/liuyx7/project/MedFlamingo-mini/examples/output1.png")
    prediction_caption = model.generate_captions(processor, images=[image])
    #annotation = open("/Users/sookim/flamingo-pytorch/roco_data/captions_roco_validation.json","r")
    #annotation = json.load(annotation)
    gt = "X-ray hand showing only two carpal bones"
    
    print("Prediction :", prediction_caption[0])
    print(prediction_caption)
    print("Ground Truth :", gt)

eval()
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

import json
import matplotlib.pyplot as plt
import torch
from flamingo_mini.utils import load_url
from PIL import Image
from IPython.display import Image as Image_display
from transformers.configuration_utils import PretrainedConfig
from IPython import embed

model = FlamingoModel.from_pretrained("/public/bme/home/liuyx7/project/MedFlamingo-mini/training/flamingo-coco/checkpoint-2000")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model.to(device)
model.eval()

processor = FlamingoProcessor(model.config)


def eval():
    image=Image.open("/public/bme/home/liuyx7/data/chex_data/p10/p10000935/s51178377/3be619d1-506a66cf-ff1ab8a1-2efb77bb-fe7d59fc.jpg")

    prediction_caption = model.generate_captions(processor, images=[image])
    #annotation = open("/Users/sookim/flamingo-pytorch/roco_data/captions_roco_validation.json","r")
    #annotation = json.load(annotation)

    print(prediction_caption[0])

eval()

'''
The lungs are clear without focal consolidation. 
No pleural effusion or pneumothorax is seen. The cardiac silhouette is top-normal. 
The mediastinal contours are unremarkable. No acute cardiopulmonary process.']
'''
'''
The cardiomediastinal and hilar contours are within normal limits. 
The lungs are well expanded. 
There is a retrocardiac opacity which is confirmed on the lateral views. 
There is no focal consolidation, pleural effusion or pneumothorax. 
There is no acute osseous abnormality. Left lower lobe pneumonia. 
Follow up radiographs after treatment are recommended to ensure resolution of this finding.
'''

'''
In comparison with the study of _ _ _, the monitoring and support devices are unchanged. 
Opacification at the left base is consistent with layering effusion and compressive atelectasis at the base. 
On the right, there is a peripheral opacification at the base that is unchanged and may have both pleural and parenchymal components.
'''
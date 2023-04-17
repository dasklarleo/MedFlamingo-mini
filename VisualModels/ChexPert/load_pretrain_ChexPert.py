import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile
from pathlib import Path
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('/home/leosher/桌面/project/MedFlamingo-mini/VisualModels/ChexPert')
from .model.classifier import Classifier  # noqa
from .utils.misc import lr_schedule  # noqa
from .model.utils import get_optimizer  # noqa

def load_pretrained_model(model_path,cfg_path):
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    model = Classifier(cfg)

    model = model.to('cuda:0').eval()
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(ckpt)
    return model

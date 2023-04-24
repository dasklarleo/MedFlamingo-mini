"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import random

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
import sys
import transformers
from transformers import HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup
sys.path.append('/public/bme/home/liuyx7/project/MedFlamingo-mini')
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor
from transformers import CLIPImageProcessor
from IPython import embed
from eval import evaluate_image_captioning  # don't ask me why this import works

import chex_dataset
from chex_dataset import ChexCaptions
logger = logging.getLogger(__name__)



chex_image_dir = '/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'

class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor()

    def __init__(self, clip_model_type: str):
        self.vision_processor =  CLIPImageProcessor()

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values']

        
def prepare_dataset(config: FlamingoConfig, json_file:str):
    transform = T.Compose([
        # add your favorite transforms
        T.Resize((224,224)),
        T.RandomHorizontalFlip(), 
        T.ToTensor()
    ])

    def target_transform(captions):
        return f"{random.choice(['', ' '])}<image>{random.choice(captions)}</s>"
    data=ChexCaptions(
        chex_image_dir, 
        json_file,
        transform=transform,
        target_transform=target_transform
    )
    return data




class DataCollator:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        for i in range(len(batch)):
            batch[i] = list (batch[i])
        pixel_values, sentences = zip(*batch) #[[[img-1.1,img-1.2]],[img-2.1]...],  [senten-1, senten-2]
        inputs = self.processor(text=sentences)
        pixel_values = torch.stack(pixel_values)
        return dict(
            pixel_values=pixel_values,
            labels=inputs['input_ids'],
            **inputs
        )


@dataclass
class FlamingoTrainingArguments(TrainingArguments):
    """ custom arguments """
    eval_coco_captioning_start: int = field(default=0)
    eval_coco_captioning_end: int = field(default=1000)
    

class FlamingoTrainer(Trainer):

    args: FlamingoTrainingArguments
    model: FlamingoModel
    processor: FlamingoProcessor
    eval_dataset: Dataset
    
    def evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """ override evaluation method to inject custom behavior. 
        """
        metrics ='TO BE DONE'

        return metrics
    
    
if __name__ == '__main__':
    # os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser(FlamingoTrainingArguments)
    training_args: FlamingoTrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s', 
        datefmt='%H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
    config = FlamingoConfig(
        clip_model_type='',
        lm='microsoft/biogpt',
        dim=1024,
        dim_visual=512,
        xattn_act='sqrelu',
        resampler_act='sqrelu'
    )
    model = FlamingoModel(config)
    model.train()

    #################################################################
    # datasets
    #################################################################
    logger.info('loading datasets...')
    train_dataset = prepare_dataset(config,'/public/bme/home/liuyx7/data/chex_data/train.json')
    test_dataset = prepare_dataset(config,'/public/bme/home/liuyx7/data/chex_data/test.json') 
    #################################################################
    # optimizer, scheduler, trainer
    #################################################################


    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(config),
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    if training_args.resume_from_checkpoint is not None:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()
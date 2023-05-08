"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import CLIPImageProcessor, HfArgumentParser
from transformers.optimization import get_constant_schedule_with_warmup
from transformers.trainer import Trainer, TrainingArguments

sys.path.append('/public/bme/home/liuyx7/project/MedFlamingo-mini')
import chex_dataset
from chex_dataset import ChexCaptions
from eval import \
    evaluate_image_captioning  
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor
from IPython import embed
from transformers import CLIPImageProcessor

logger = logging.getLogger(__name__)
os.environ["WANDB_MODE"] = "offline"


chex_image_dir = '/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'

class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor()

    def __init__(self, clip_model_type: str):
        self.vision_processor =  CLIPImageProcessor()

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values']

        
def prepare_dataset(config: FlamingoConfig, data_path:str):
    transform = T.Compose([
        # add your favorite transforms
        T.Resize((224,224)),
        # T.RandomHorizontalFlip(), TODO Add the horizontaoFlip in both image and captions
        T.ToTensor()
    ])

    def target_transform(captions):
        return f"{random.choice(['', ' '])}<Diagnosis report of chest X-ray>{random.choice(captions)}</s>" # Each image group may contain several captions
    data=ChexCaptions(
        chex_image_dir,
        data_path,
        transform=transform,
        target_transform=target_transform,
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
    captioning_prefix: str = field(default="<Diagnosis report of chest X-ray>")         # It's a common thing to do for COCO image captioning
    captioning_start: int = field(default=0)
    #captioning_end: int = field(default=1000)
    
    num_train_epochs: float = field(default=10)    

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
        """
        override evaluation method to inject custom behavior. 
        """
        metrics = evaluate_image_captioning(self.eval_dataset, self.model, 
            prefix=self.args.captioning_prefix,
            start=self.args.captioning_start,
            end=None,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers
        )
        
        metrics = {f"{metric_key_prefix}_{k}" : v for k, v in metrics.items()}

        # HF trainer stuff from overridden method
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
    
if __name__ == '__main__':
    # os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
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
    config = FlamingoConfig( #detailed in file configuration_flamingo.py. This config is passed to the FlamingoModel in modeling_flamingo.py
        clip_model_type='',
        lm='microsoft/biogpt',
        dim=1024,
        dim_visual=512,
        xattn_act='sqrelu',
        resampler_act='sqrelu',
        freeze_vision_model=False
    )
    model = FlamingoModel(config)
    model.train()

    #################################################################
    # datasets
    #################################################################
    logger.info('loading datasets...')
    train_dataset = prepare_dataset(config,'/public/bme/home/liuyx7/data/chex_data/mimic.pkl')
    eval_dataset = prepare_dataset(config,'/public/bme/home/liuyx7/data/chex_data/mimic_test_100.pkl') 
    ##  ###############################################################
    # optimizer, scheduler, trainer
    #################################################################


    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        data_collator= DataCollator(config),
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    
    trainer.train()
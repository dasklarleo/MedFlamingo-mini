from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader

from flamingo_mini import FlamingoModel, FlamingoProcessor
import sys
sys.path.append('/public/bme/home/liuyx7/project/MedFlamingo-mini/metrics') 
from bleu import Bleu

class MyDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, reference = self.dataset[index]
        image_id = self.dataset.ids[index]
        return image_id, image, reference


@torch.no_grad()
def evaluate_image_captioning(
    dataset,
    model: FlamingoModel, 
    *,
    prefix: str = "<Report of chest X-ray image>",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 4, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    results = []
    ground_truths = []
    
    wrapper = MyDatasetWrapper(dataset)
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers)

    for image_ids, pixels, references in tqdm(loader, disable=not verbose):
        captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix
        )
        for c in captions:
            results.append(c)
        for r in references:
            ground_truths.append([r])
    bleu_metrics = Bleu()
    
    return bleu_metrics.compute(predictions = results, references = ground_truths)


import os.path
from typing import Any, Callable, List, Optional, Tuple
import sys
from PIL import Image
#sys.path.append('/home/leosher/桌面/project/MedFlamingo-mini/training/')
from torchvision.datasets.vision import VisionDataset
from chex_class import Chex


class ChexPert(VisionDataset):
    """
    Args:
        root (string): Root directory where images are saved to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = Chex(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))

    def _load_image(self, id: int) -> Image.Image:
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        i = 0
        # TODO Add multiple images
        

        return  Image.open(file_name).convert("RGB")#[ H, W, 3]
    
    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(id)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id) # TODO Add different images
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class ChexCaptions(ChexPert):


    def _load_target(self, id: int) -> List[str]:
        return [ann["annotations"] for ann in super()._load_target(id)]

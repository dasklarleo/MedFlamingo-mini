import sys
from typing import Any, Callable, List, Optional, Tuple

from chex_class import Chex
from PIL import Image
#sys.path.append('/home/leosher/桌面/project/MedFlamingo-mini/training/')
from torchvision.datasets.vision import VisionDataset


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
        dataset_path:str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.chex = Chex(dataset_path)
        self.ids = list(sorted(self.chex.anns.keys()))

    def _load_image(self, id: int) -> Image.Image:
        file_name = self.chex.loadImgs(id)[0]["file_name"]
        i = 0
        # TODO Add multiple images
        

        return  Image.open(file_name).convert("RGB")#[ H, W, 3]
    
    def _load_target(self, id: int) -> List[Any]:
        return self.chex.loadAnns(id)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        images = self._load_image(id) # TODO Add different images
        target = self._load_target(id)

        if self.transforms is not None:
            images, target = self.transforms(images, target)

        return images, target

    def __len__(self) -> int:
        return len(self.ids)


class ChexCaptions(ChexPert):


    def _load_target(self, id: int) -> List[str]:
        return [ann["annotations"] for ann in super()._load_target(id)]

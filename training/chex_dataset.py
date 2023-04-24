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
        file_names = self.coco.loadImgs(id)[0]["file_name"]
        images=[]
        i = 0
        for file in file_names:
            images.append(Image.open(file).convert("RGB"))
            i+=1
        if i<3:
            while i<3:
                i+=1
                images.append(Image.open(file_names[0]).convert("RGB"))
        return images #[N, H, W, 3]
    
    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(id)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class ChexCaptions(ChexPert):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.PILToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id: int) -> List[str]:
        return [ann["annotations"] for ann in super()._load_target(id)]

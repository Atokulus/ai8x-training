import os
from xmlrpc.client import Boolean
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import ai8x
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.utils import save_image
from torchvision.transforms import Resize

WIDTH = 384
HEIGHT = 384
FOLD_RATIO = 2

class WeedDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, transform=None, platform_transform=None, train: Boolean = False
    ):
        self.train = train
        split = 5

        self.transform = transform
        self.platform_transform = platform_transform
        # load all image files, sorting them to
        # ensure that they are aligned
        dataset_folder = Path(root) / "weed"

        self.image_folder: Path = dataset_folder / "images"
        self.annotation_folder: Path = dataset_folder / "annotations"
        self.mask_path: Path = dataset_folder / "masks"

        self.imgs = list(sorted(self.image_folder.glob("*.png")))
        self.annotations = list(sorted(self.annotation_folder.glob("*.png")))
        self.masks = list(sorted(self.mask_path.glob("*.png")))

        if train:
            self.imgs = self.imgs[split:-1]
            self.annotations = self.annotations[split:-1]
            self.masks = self.masks[split:-1]
        else:
            self.imgs = self.imgs[0:split]
            self.annotations = self.annotations[0:split]
            self.masks = self.masks[0:split]

        first_image = Image.open(self.imgs[0])

        self.new_width = first_image.width
        self.new_height = first_image.height

        self.image_data = [np.array(Image.open(img_path).convert("RGB").resize((self.new_width, self.new_height))).astype('float32') for img_path in self.imgs]
        self.image_data = [img / np.amax(img) for img in self.image_data]
        self.annotation_data = [np.array(Image.open(img_path).convert("RGB").resize((self.new_width, self.new_height))) for img_path in self.annotations]
        
        zero_image = np.zeros((self.new_height, self.new_width), dtype=np.uint8)

        crop_annotation = [np.greater(annotation[..., 0], 0).astype(np.uint8) for annotation in self.annotation_data]
        weed_annotation = [np.greater(annotation[..., 1], 0).astype(np.uint8) for annotation in self.annotation_data]

        self.annotation_data = [np.clip(zero_image + crop + 2 * weed, 0, 2) for (crop, weed) in zip(crop_annotation, weed_annotation)]



    def __getitem__(self, idx):
        img = self.image_data[idx]
        annotation = self.annotation_data[idx]

        if self.transform is not None:
            while True:
                transformed = self.transform(image=img, annotation=annotation)
                transformed_image = transformed["image"]
                transformed_mapped_annotation = transformed["annotation"]

                # Enforce weed with higher probability in test set
                if torch.max(transformed_mapped_annotation) < 2 and np.random.rand(1) > 0.1:
                    continue
                else:
                    break

        else:
            transformed_image = torch.as_tensor(img, dtype=torch.double)
            transformed_mapped_annotation = torch.as_tensor(annotation, dtype=torch.uint8)

        transformed_image = Resize(size=(int(HEIGHT / FOLD_RATIO), int(WIDTH / FOLD_RATIO)))(transformed_image)
        transformed_mapped_annotation = transformed_mapped_annotation.view(HEIGHT, WIDTH).long()

        if self.platform_transform is not None:
            transformed_image = self.platform_transform(transformed_image)

        #save_image(transformed_image.clone() / 2 + 0.5, f"input_{idx}.png")
        #save_image(transformed_mapped_annotation.clone() / 2, f"target_{idx}.png")

        return transformed_image, transformed_mapped_annotation

    def __len__(self):
        return len(self.imgs)


def weed_get_datasets(data, load_train=True, load_test=True):
    """
    Load the Weed dataset.
    """
    (data_dir, args) = data

    if load_train:
        train_transform = A.Compose(
            transforms=[
                A.RandomScale(),
                A.RandomCrop(WIDTH, HEIGHT),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                ToTensorV2(),
            ],
            additional_targets={"annotation": "image"},
        )

        ai8x_transform = transforms.Compose([ai8x.normalize(args=args)])

        train_dataset = WeedDataset(
            root=data_dir,
            train=True,
            transform=train_transform,
            platform_transform=ai8x_transform,
        )
    else:
        train_dataset = None

    if load_test:
        test_transform = A.Compose(
            transforms=[
                A.RandomScale(),
                A.RandomCrop(WIDTH, HEIGHT),
                ToTensorV2(),
            ],
            additional_targets={"annotation": "image"},
        )

        ai8x_transform = transforms.Compose([ai8x.normalize(args=args)])

        test_dataset = WeedDataset(
            root=data_dir,
            train=False,
            transform=test_transform,
            platform_transform=ai8x_transform,
        )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        "name": "Weed",
        "input": (3, 512, 512),
        "output": (0, 1, 2),
        "loader": weed_get_datasets,
    },
]

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


class WeedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, platform_transform=None, train: Boolean = False):
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

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        annotation_path = self.annotations[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        annotation = Image.open(annotation_path).convert("RGB")
        width = annotation.width
        height = annotation.height
        annotation = np.array(annotation)

        crop_annotation = np.greater(annotation[..., 1], 0) * 1
        weed_annotation = np.greater(annotation[..., 0], 0) * 1

        mapped_annotation = (
            np.ones((height, width), dtype=np.uint8) * 0.5
            - crop_annotation
            + weed_annotation
        )

        if self.transform is not None:
            transformed = self.transform(image=img, annotation=mapped_annotation)
            img = transformed["image"]
            mapped_annotation = transformed["annotation"]

        if self.platform_transform is not None:
            img = self.platform_transform(img)
            mapped_annotation = self.platform_transform(mapped_annotation)

        # img = torch.as_tensor(img, dtype=torch.int32)
        # mapped_annotation = torch.as_tensor(mapped_annotation, dtype=torch.uint8)

        return img, mapped_annotation

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
                A.RandomCrop(128, 128),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
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
            transforms=[A.RandomCrop(128, 128), ToTensorV2()],
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
        "input": (3, 128, 128),
        "output": (0, 1, 2),
        "loader": weed_get_datasets,
    },
]

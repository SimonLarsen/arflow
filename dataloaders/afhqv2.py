import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.transforms.functional import (
    crop,
    resize,
    hflip,
    pil_to_tensor,
)


class AFHQv2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        target_size: int = 256,
        do_normalize: bool = True,
        do_augment: bool = False,
        crop_min_ratio: float = 0.9,
        num_val: int = 256,
        p_drop: float = 0.1,
    ):
        super().__init__()

        assert split in ("train", "val")

        self.dataset = load_dataset("huggan/AFHQv2")["train"]

        val_rows = (
            np.linspace(0, len(self.dataset) - 1, num_val).round().astype(int)
        )
        train_rows = set(range(len(self.dataset))).difference(val_rows)
        if split == "train":
            self.dataset = self.dataset.select(train_rows)
        else:
            self.dataset = self.dataset.select(val_rows)

        self.target_size = target_size
        self.do_normalize = do_normalize
        self.do_augment = do_augment
        self.crop_min_ratio = crop_min_ratio
        self.p_drop = p_drop

    def __len__(self) -> int:
        return len(self.dataset)

    def normalize(self, x):
        return x * 2 - 1

    def denormalize(self, x):
        return x / 2 + 0.5

    def augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = hflip(x)
        return x

    def fit_image(self, x):
        h, w = x.shape[-2:]

        if self.do_augment:
            crop_max = min(h, w)
            crop_min = round(crop_max * self.crop_min_ratio)
            crop_size = torch.randint(crop_min, crop_max + 1, (1,)).item()
            left = torch.randint(0, w - crop_size + 1, (1,)).item()
            top = torch.randint(0, h - crop_size + 1, (1,)).item()
        else:
            crop_size = min(h, w)
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        x = crop(x, top, left, crop_size, crop_size)
        x = resize(x, self.target_size, antialias=True)
        return x

    def __getitem__(self, idx: int):
        entry = self.dataset[idx]
        image = entry["image"]
        label = entry["label"]

        image = pil_to_tensor(image) / 255.0
        if image.size(0) == 4:
            rgb = image[:3]
            alpha = image[3:]
            image = alpha * rgb + (1.0 - alpha) * torch.ones_like(rgb)

        image = self.fit_image(image)

        if self.do_augment:
            image = self.augment(image)

        if self.do_normalize:
            image = self.normalize(image)

        if self.p_drop > 0.0:
            if torch.rand(1).item() < self.p_drop:
                label = 0
            else:
                label += 1

        return (image, label), image

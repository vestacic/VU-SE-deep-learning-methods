import math
from pathlib import Path
import random
from typing import Literal

from PIL import Image
import torch
import torchvision


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        classes: list[str],
        root_directory: Path,
        mode: Literal["train", "test", "validation"],
        reverse_colours: dict[tuple[int, int, int], int],
        normalization_mean: list[float],
        normalization_std: list[float],
    ) -> None:
        self.root_directory = root_directory
        self.mode = mode
        self.images = []
        for class_name in classes:
            self.images.extend(
                (root_directory / mode / class_name / "data").rglob("*")
            )
        self.images = sorted(self.images)
        self.reverse_colours = reverse_colours
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = (
            self.root_directory
            / self.mode
            / image_path.parents[1].name
            / "coloured_labels"
            / image_path.name
        ).with_suffix(".png")

        image = Image.open(fp=image_path).convert("RGB")
        mask = Image.open(fp=mask_path).convert("RGB")

        return self.transform(image=image, mask=mask)

    def transform(
        self, image: Image, mask: Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "train":
            if random.random() > 0.5:
                image = torchvision.transforms.functional.hflip(img=image)
                mask = torchvision.transforms.functional.hflip(img=mask)

            width, height = image.size
            pad_width = max(0, 256 - width)
            pad_height = max(0, 256 - height)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top

            if (pad_left + pad_right) > 0 or (pad_top + pad_bottom) > 0:
                image = torchvision.transforms.functional.pad(
                    img=image,
                    padding=(pad_left, pad_top, pad_right, pad_bottom),
                    fill=0,
                    padding_mode="constant",
                )
                mask = torchvision.transforms.functional.pad(
                    img=mask,
                    padding=(pad_left, pad_top, pad_right, pad_bottom),
                    fill=0,
                    padding_mode="constant",
                )

            crop_params = torchvision.transforms.RandomCrop.get_params(
                image, output_size=(256, 256)
            )
            image = torchvision.transforms.functional.crop(image, *crop_params)
            mask = torchvision.transforms.functional.crop(mask, *crop_params)

            rotation_angle = random.choice([i * 15 for i in range(0, 24)])
            image = torchvision.transforms.functional.rotate(
                img=image, angle=rotation_angle
            )
            mask = torchvision.transforms.functional.rotate(
                img=mask, angle=rotation_angle
            )

            image = torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )(image)

        else:
            pad_x = math.ceil(image.width / 8) * 8 - image.width
            pad_y = math.ceil(image.height / 8) * 8 - image.height
            image = torchvision.transforms.functional.pad(
                image, [0, 0, pad_x, pad_y]
            )
            mask = torchvision.transforms.functional.pad(
                mask, [0, 0, pad_x, pad_y]
            )

        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = torchvision.transforms.Normalize(
            mean=self.normalization_mean, std=self.normalization_std
        )(image_tensor)
        mask_tensor = self.mask_to_tensor(mask=mask)
        return (image_tensor, mask_tensor)

    def mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask = mask.convert("RGB")

        mask_np = torch.ByteTensor(
            torch.ByteStorage.from_buffer(mask.tobytes())
        )
        mask_np = mask_np.view(mask.size[1], mask.size[0], 3)

        height, width, _ = mask_np.shape
        out_tensor = torch.zeros((height, width), dtype=torch.long)

        for (r, g, b), class_index in self.reverse_colours.items():
            match = (
                (mask_np[:, :, 0] == r)
                & (mask_np[:, :, 1] == g)
                & (mask_np[:, :, 2] == b)
            )
            out_tensor[match] = class_index

        return out_tensor

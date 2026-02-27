# -*- coding:utf-8 -*-
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageOps
import numpy as np
import pickle
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
import os

class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)

class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

class MultiViewDataInjector():
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample, mask):
        output, mask, box = zip(*[transform(sample, mask) for transform in self.transform_list])
        output_cat = torch.stack(output, dim=0)
        mask_cat = torch.stack(mask)

        return output_cat, mask_cat, box


class SSLMaskDataset(VisionDataset):
    def __init__(self, root: str, mask_file: str, extensions=IMG_EXTENSIONS, transform=None, subset=""):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.img_to_mask = self._get_masks(mask_file)
        self.loader = default_loader
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.solarize_prob = 0
        self.gb_prob=1.0
        self.com_img_transform = transforms.Compose([
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=self.gb_prob),
            transforms.RandomApply([Solarize()], p=self.solarize_prob),
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def _get_masks(self, mask_file):
        with open(mask_file, "rb") as file:
            return pickle.load(file)

    def _get_intersection(self, box1, box2):
        """
        Compute the intersection of two boxes.
        box1, box2 are assumed to be in the format (x1, y1, x2, y2), i.e., top-left and bottom-right coordinates.
        Returns the intersecting box coordinates.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # If there is an intersection, return the intersecting box; otherwise, return None
        if x1 < x2 and y1 < y2:
            return (x1, y1, x2, y2)
        else:
            return None

    def __getitem__(self, index: int):
        img_path, mask_path = self.img_to_mask[index]

        # Load Image
        img = self.loader(img_path)

        # Load Mask
        with open(mask_path, "rb") as file:
            mask = pickle.load(file)

        # Apply transforms (get two boxes)
        if self.transform is not None:
            img_transform, mask, boxes = self.transform(img, mask.unsqueeze(0))

        # Assuming boxes contain two boxes: box1 and box2
        box1, box2 = boxes[0], boxes[1]  # Assuming boxes are in a list

        # Compute intersection box
        intersect_box = self._get_intersection(box1, box2)

        if intersect_box:
            # If there is an intersection, crop the common area from the image
            x1, y1, x2, y2 = intersect_box
            com_img = np.array(img)[y1:y2, x1:x2]  # Crop the image
            com_img = Image.fromarray(com_img)
            # Apply a transformation to the cropped common area
            com_img = self.com_img_transform(com_img)  # Apply your specific transformation here

            return img_transform, com_img, mask
        else:
            # If no intersection, create a zero-filled image with the same size as img
            com_img = torch.zeros_like(img_transform[0])  # Zero-filled tensor with the same size as img_transform
            return img_transform, com_img, mask

    def __len__(self) -> int:
        return len(self.img_to_mask)

class CustomCompose:
    def __init__(self, t_list, p_list):
        self.t_list = t_list
        self.p_list = p_list

    def __call__(self, img, mask):
        for p in self.p_list:
            img, mask, crop_box = p(img, mask)
        for t in self.t_list:
            img = t(img)
        return img, mask, crop_box

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.t_list:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class MaskRandomResizedCrop():
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image or Tensor): Image to be cropped and resized.
            mask (Tensor): Mask to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped/resized image.
            Mask Tensor: Randomly cropped/resized mask.
        """
        # import ipdb;ipdb.set_trace()
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
        image = transforms.functional.resize(transforms.functional.crop(image, i, j, h, w), (self.size, self.size),
                                             interpolation=transforms.functional.InterpolationMode.BICUBIC)

        image = self.topil(torch.clip(self.totensor(image), min=0, max=255))
        mask = transforms.functional.resize(transforms.functional.crop(mask, i, j, h, w), (self.size, self.size),
                                            interpolation=transforms.functional.InterpolationMode.NEAREST)

        return [image, mask, [i, j, h, w]]


class MaskRandomHorizontalFlip():
    """
    Apply horizontal flip to a PIL Image and Mask.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image or Tensor): Image to be flipped.
            mask (Tensor): Mask to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
            Mask Tensor: Randomly flipped mask.
        """

        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            return [image, mask, []]
        return [image, mask, []]


def get_transform(stage, gb_prob=1.0, solarize_prob=0., crop_size=224):
    t_list = []
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if stage in ('train', 'val'):
        t_list = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_prob),
            transforms.RandomApply([Solarize()], p=solarize_prob),
            transforms.ToTensor(),
            normalize]

        p_list = [
            MaskRandomHorizontalFlip(),
            MaskRandomResizedCrop(crop_size),
        ]

    elif stage == 'ft':
        t_list = [
            transforms.ToTensor(),
            normalize]

        p_list = [
            MaskRandomHorizontalFlip(),
            MaskRandomResizedCrop(crop_size),
        ]

    elif stage == 'test':
        t_list = [
            transforms.ToTensor(),
            normalize]

        p_list = [
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
        ]

    transform = CustomCompose(t_list, p_list)
    return transform

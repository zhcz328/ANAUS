#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor
from utils.mask_utils import convert_binary_mask
import numpy as np
from scipy.special import comb

class AppearanceTransform2D:
    def __init__(self, local_rate=0.8, nonlinear_rate=0.9, inpaint_rate=0.2, is_local=True, is_nonlinear=True, is_in_painting=True):
        self.is_local = is_local
        self.is_nonlinear = is_nonlinear
        self.is_in_painting = is_in_painting
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate
        self.inpaint_rate = inpaint_rate

    def rand_aug(self, data):
        """
        Apply augmentation to each image in the batch.
        """
        return torch.stack([self._augment_single_image(img) for img in data])

    def _augment_single_image(self, x):
        """
        Apply augmentations to a single image.
        """
        if self.is_local:
            x = self.local_pixel_shuffling(x, prob=self.local_rate)
        if self.is_nonlinear:
            x = self.nonlinear_transformation(x, self.nonlinear_rate)
        if self.is_in_painting:
            x = self.image_in_painting(x)
        return x

    def nonlinear_transformation(self, x, prob=0.5):
        if torch.rand(1).item() >= prob:
            return x
        points = [[0, 0], [torch.rand(1).item(), torch.rand(1).item()], [torch.rand(1).item(), torch.rand(1).item()], [1, 1]]
        xvals, yvals = self.bezier_curve(points, nTimes=1000)
        xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x.cpu().numpy(), xvals, yvals)
        return torch.from_numpy(nonlinear_x).to(x.device)

    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def local_pixel_shuffling(self, x, prob=0.5):
        if torch.rand(1).item() >= prob:
            return x
        img = x.clone()
        _, img_rows, img_cols = img.shape
        block_size = max(5, img_rows // 20)  # Dynamically resize blocks
        num_block = (img_rows * img_cols) // (block_size ** 2)  # Adjust the number of blocks based on the size of the image

        for _ in range(num_block):
            x1 = torch.randint(0, img_rows - block_size + 1, (1,)).item()
            y1 = torch.randint(0, img_cols - block_size + 1, (1,)).item()
            block = img[:, x1:x1 + block_size, y1:y1 + block_size]
            for c in range(block.size(0)):  # Scramble each channel independently
                channel_block = block[c].clone().view(-1)
                channel_block = channel_block[torch.randperm(channel_block.size(0))].view(block_size, block_size)
                block[c] = channel_block
        return img

    def image_in_painting(self, x):
        _, img_rows, img_cols = x.shape
        for _ in range(10):
            if torch.rand(1).item() < 0.95:
                block_size_x = torch.randint(img_rows // 10, img_rows // 5, (1,)).item()
                block_size_y = torch.randint(img_cols // 10, img_cols // 5, (1,)).item()
                x1, y1 = torch.randint(0, img_rows - block_size_x, (1,)).item(), torch.randint(0, img_cols - block_size_y, (1,)).item()
                x[:, x1:x1 + block_size_x, y1:y1 + block_size_y] = torch.rand(block_size_x, block_size_y)
        return x

    def image_out_painting(self, x):
        img = torch.rand_like(x)
        _, img_rows, img_cols = x.shape
        block_size_x = torch.randint(3 * img_rows // 7, 4 * img_rows // 7, (1,)).item()
        block_size_y = torch.randint(3 * img_cols // 7, 4 * img_cols // 7, (1,)).item()
        x1, y1 = torch.randint(0, img_rows - block_size_x, (1,)).item(), torch.randint(0, img_cols - block_size_y, (1,)).item()
        img[:, x1:x1 + block_size_x, y1:y1 + block_size_y] = x[:, x1:x1 + block_size_x, y1:y1 + block_size_y]
        return img

class ANAUSModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']
        self.transform = AppearanceTransform2D(local_rate=0.8, nonlinear_rate=0.9, inpaint_rate=0.2)

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)

        # predictor
        self.predictor = Predictor(config)

        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    def forward(self, view1, view2, com_images, mm, masks):
        # online network forward

        masks = torch.cat([masks[:, i, :, :, :] for i in range(masks.shape[1])])

        result = []
        result_com = []
        for pool_size in [7,14,28,56]:
            # print(pool_size)
            masks_use = convert_binary_mask(masks, pool_size=pool_size)
            q, pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0), masks_use, 'online', pool_size))

            # target network forward
            with torch.no_grad():
                self._update_target_network(mm)
                target_z, tinds = self.target_network(torch.cat([view1, view2], dim=0), masks_use, 'target', pool_size)
                target_z = target_z.detach().clone()
            result.append([q, target_z, pinds, tinds])
            damage_com_images = self.transform.rand_aug(com_images).float()
            q_com, _ = self.predictor(
                *self.online_network(torch.cat([damage_com_images, damage_com_images], dim=0), masks_use, 'online',
                                     pool_size))
            target_z_com, _ = self.target_network(torch.cat([com_images, com_images], dim=0), masks_use, 'target',
                                                  pool_size)
            result_com.append([q_com, target_z_com])

        return result, result_com

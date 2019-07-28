from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from config import Config
import utils

transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    transforms.CenterCrop(Config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def load_image_dataloader(root_dir, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True):
    """
    :param root_dir: directory that contains another directory of images. All images should be under root_dir/<some_dir>/
    :param batch_size: batch size
    :param num_workers: number of workers for torch.utils.data.DataLoader
    :param shuffle: use shuffle
    :return: torch.utils.Dataloader object
    """
    assert os.path.isdir(root_dir)

    image_dataset = ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(image_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader


def load_style_image(image_path):
    style_image = utils.load_image(image_path, Config.style_image_size)
    style_image_tensor = style_transform(style_image)
    return style_image_tensor

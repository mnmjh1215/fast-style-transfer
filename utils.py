# all utils are here

import torch
from PIL import Image


def load_image(filename, size=None):
    img = Image.open(filename)
    if size:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(tensor):
    batch, channels, h, w = tensor.shape
    features = tensor.view(batch, channels, -1)
    gram = torch.bmm(features, features.transpose(1, 2)) / (channels * h * w)
    return gram


def normalize_generated_image(tensor):
    tensor = tensor / 255.0
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (tensor - mean) / std
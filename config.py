# all configurations are saved here
import torch


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 256
    batch_size = 4
    num_workers = 2
    style_image_size = None
    style_weight = 1e12
    content_weight = 1e5
    num_epochs = 2
    lr = 5e-4

    content_path = 'data/content/'
    style_path = 'data/style/style.jpg'

    model_save_path = 'checkpoints/'
    test_image_path = 'data/test/'
    image_save_path = 'generated-images/'

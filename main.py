

from model import TransformationNetwork, FeatureExtractor
from train import Trainer
from utils import save_image
from dataloader import load_image_dataloader, load_style_image
from config import Config

import argparse
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='Use this argument to test generator')

    parser.add_argument('--model_path',
                        help='Path to saved model')

    parser.add_argument('--content_image_path',
                        default=Config.content_path,
                        help='Path to content images')

    parser.add_argument('--style_image_path',
                        default=Config.style_path,
                        help='Path to style image')

    parser.add_argument('--model_save_path',
                        default=Config.model_save_path,
                        help='Path to save trained model')

    parser.add_argument('--test_image_path',
                        default=Config.test_image_path,
                        help='Path to test photo images, required if testing')

    parser.add_argument('--image_save_path',
                        default=Config.image_save_path)

    parser.add_argument('--num_epochs',
                        type=int,
                        default=Config.num_epochs,
                        help='Number of training epochs')

    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='Learning rate')

    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')

    args = parser.parse_args()

    return args


def load_network(network, model_path):
    ckpt = torch.load(model_path, map_location=Config.device)
    network.load_state_dict(ckpt['network_state_dict'])


def generate_and_save_images(network, test_image_loader, save_path):
    network.eval()
    tensor_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    image_ix = 0
    for test_images, _ in test_image_loader:
        test_images = test_images.to(Config.device)
        generated_images = network(test_images).detach().cpu()

        for i in range(len(generated_images)):
            image = generated_images[i]
            image = tensor_to_image(image)
            image.save(os.path.join(save_path, '{0}.jpg'.format(image_ix)))
            image_ix += 1


def main():
    args = get_args()

    device = Config.device
    print("PyTorch running with device {0}".format(device))

    print("Creating transformation network...")
    network = TransformationNetwork().to(device)

    if args.test:
        assert args.model_path, '--model_path must be provided for testing'
        print('Testing...')
        network.eval()

        print("Loading transformation network...")
        load_network(network, args.model_path)

        test_image_loader = load_image_dataloader(args.test_image_path, batch_size=Config.batch_size * 2, shuffle=False)
        if not os.path.isdir(args.image_save_path):
            os.makedirs(args.image_save_path)

        print("Generating images...")
        generate_and_save_images(network, test_image_loader, args.image_save_path)

    else:
        # Training!
        print("Training...")

        print("Loading feature extractor")
        feature_extractor = FeatureExtractor()

        # load dataloader
        print("Loading data")
        content_image_dataloader = load_image_dataloader(args.content_image_path, batch_size=args.batch_size)
        style_image = load_style_image(args.style_image_path)

        print("Loading Trainer")
        trainer = Trainer(network, feature_extractor, content_image_dataloader, style_image, lr=args.lr)
        if args.model_path:
            trainer.load_checkpoint(args.model_path)

        print("Start training")
        loss_hist = trainer.train(num_epochs=args.num_epochs)

        if not os.path.isdir(args.model_save_path):
            os.makedirs(args.model_save_path)
        trainer.save_checkpoint(os.path.join(args.model_save_path, 'epoch-{0}.ckpt'.format(args.num_epochs)))

        plt.plot(loss_hist, label='Loss')
        plt.legend()
        plt.savefig('StyleTransfer_train_history.jpg')
        plt.show()


if __name__== '__main__':
    main()
from utils import gram_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import normalize_generated_image

from config import Config


class Trainer:
    def __init__(self, transformation_network, feature_extractor,
                 train_dataloader, style_image_tensor,
                 content_weight=Config.content_weight, style_weight=Config.style_weight,
                 lr=Config.lr, print_every=100):

        self.network = transformation_network.to(Config.device)
        self.feature_extractor = feature_extractor.to(Config.device)

        self.train_dataloader = train_dataloader
        self.style_image_tensor = style_image_tensor

        self.content_weight = content_weight
        self.style_weight = style_weight

        self.mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.curr_epoch = 0
        self.loss_hist = []
        self.print_every = print_every

        # pre-calculate gram matrix of style image
        style_image_tensor = style_image_tensor.repeat(train_dataloader.batch_size, 1, 1, 1).to(Config.device)  # batch_size x c x h x w
        style_features = self.feature_extractor(style_image_tensor)
        self.style_gram_matrix = dict([(key, gram_matrix(style_features[key])) for key in style_features])

    def train(self, num_epochs):
        print("Start training for {0} epochs".format(num_epochs))
        for epoch in range(self.curr_epoch, num_epochs):
            start = time.time()
            epoch_loss = 0
            for ix, (train_images, _) in enumerate(self.train_dataloader):
                train_images = train_images.to(Config.device)
                loss = self.train_step(train_images)

                self.loss_hist.append(loss)
                epoch_loss += loss

                if (ix + 1) % self.print_every == 0:
                    print("Epoch [{0}/{1}] Iteration {2} loss: {3:.4f}".format(
                        epoch + 1, num_epochs, ix + 1, epoch_loss / (ix + 1)
                    ))

            # end of epoch
            print("Epoch [{0}/{1}] loss: {2:.4f}, {3:.4f} seconds".format(
                epoch + 1, num_epochs, epoch_loss / (ix + 1), time.time() - start
            ))
            self.curr_epoch += 1

        return self.loss_hist

    def train_step(self, train_images):
        self.optimizer.zero_grad()

        batch_size = train_images.shape[0]
        out_images = self.network(train_images)
        content_features = self.feature_extractor(train_images)
        out_features = self.feature_extractor(out_images)

        content_loss = self.content_weight * self.mse(out_features['relu_2_2'], content_features['relu_2_2'])

        style_loss = 0
        out_gram = dict([(key, gram_matrix(out_features[key])) for key in out_features])
        for key in out_gram:
            style_loss += self.style_weight * self.mse(out_gram[key], self.style_gram_matrix[key][:batch_size])

        loss = content_loss + style_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_hist': self.loss_hist,
            'curr_epoch': self.curr_epoch
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=Config.device)
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.loss_hist = ckpt['loss_hist']
        self.curr_epoch = ckpt['curr_epoch']




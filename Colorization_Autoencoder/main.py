import torchvision
import cv2
import numpy as np
import torch.nn as nn
import torch as T
from torch.utils.data import DataLoader
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from encoder import encoder
from decoder import decoder
from autoencoder import autoencoder
from torchsummary import summary

batch_size = 16
epochs = 30
latent_size = 100

# T.cuda.empty_cache()
# import gc
# gc.collect()

if __name__ == "__main__":
    grayscale_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    grayscale = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=grayscale_transform)
    grayscaleloader = T.utils.data.DataLoader(grayscale, batch_size=batch_size,
                                              num_workers=2, drop_last=True)

    colored = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    coloredloader = T.utils.data.DataLoader(colored, batch_size=batch_size,
                                              num_workers=2, drop_last=True)

    en = iter(coloredloader)

    test_data_gray = torchvision.datasets.CIFAR10(root='./data', train=False,transform=grayscale_transform,
                                           download=True)

    test_loader_gray = T.utils.data.DataLoader(test_data_gray, batch_size=batch_size,
                                            num_workers=2, drop_last=True)

    test_gray = iter(test_loader_gray)

    test_data_clr = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    test_loader_clr = T.utils.data.DataLoader(test_data_clr, batch_size=batch_size,
                                            num_workers=2, drop_last=True)

    test_clr = iter(test_loader_clr)

    original, _ = next(test_clr)
    o = original[0]
    print(o.shape)
    ori = o.view(o.shape[1], o.shape[2], o.shape[0])
    print(ori.shape)
    ori = ori.cpu().detach()
    ori = np.array(ori)
    ori = cv2.resize(ori, (256,256))
    cv2.imwrite(f"generated_images/original{1}.png", 255*ori)
    cv2.imshow("", ori)
    cv2.waitKey(0)

    # encoder = encoder(latent_size, 0.01)
    # encoder = encoder.to(device)
    # #summary(encoder, (64, 1, 32, 32))
    #
    # decoder = decoder(latent_size, 0.01)
    # decoder = decoder.to(device)
    # #summary(decoder, (64, 100))

    autoencoder = autoencoder(latent_size)
    autoencoder = autoencoder.to(device)
    # for epoch in range(epochs):
    #     print("sa")
    #     en = iter(coloredloader)
    #     A_loss_total = 0
    #     for batch_idx, (images, labels) in enumerate(grayscaleloader):
    #         images = images.to(device)
    #         autoencoder.optimizer.zero_grad()
    #         colorized = autoencoder.forward(images)
    #
    #         original, _ = en.next()
    #         original = original.to(device)
    #         A_loss = autoencoder.loss(colorized, original).to(device)
    #         A_loss_total += A_loss.item()
    #         A_loss.backward()
    #
    #         A_optim = autoencoder.optimizer
    #         A_optim.step()
    #
    #     print(f"For epoch {epoch} Discriminator Loss is : {A_loss_total / len(grayscaleloader)}")
    #
    # autoencoder.save_checkpoint()
    #
    #
    autoencoder.load_checkpoint()

    # def save_images():
    #     autoencoder.eval()
    #     gray_image, _ = next(test_gray)
    #     gray_image = T.Tensor(gray_image).to(device)
    #
    #     colorized = autoencoder.forward(gray_image)
    #
    #     original, _ = next(test_clr)
    #     counter = 0
    #     for image in colorized:
    #         img = image.view(image.shape[1], image.shape[2], image.shape[0])
    #         img = img.cpu().detach()
    #         img = np.array(img)
    #
    #         gray_im = gray_image[counter].view(gray_image[counter].shape[1], gray_image[counter].shape[2], gray_image[counter].shape[0])
    #         gray_im = gray_im.cpu().detach()
    #         gray_im = np.array(gray_im)
    #
    #         ori = original[counter].view(original[counter].shape[1], original[counter].shape[2], original[counter].shape[0])
    #         ori = ori.cpu().detach()
    #         ori = np.array(ori)
    #
    #         cv2.imwrite(f"generated_images/gray{counter}.jpg", 255*gray_im)
    #         cv2.imwrite(f"generated_images/colorized{counter}.jpg", 255*img)
    #         cv2.imwrite(f"generated_images/original{counter}.jpg", 255*ori)
    #         counter += 1
    #
    # save_images()
import torchvision
import argparse
import os
import yaml
import matplotlib.pyplot as plt
import time
import sys
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from utils import *
from models import CNN6, CNN6d, FCN3
from recursive_attack import r_gap, peeling, fcn_reconstruction, inverse_udldu
from multiprocessing import Pool

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
parser = argparse.ArgumentParser(
    description="Model related arguments. For other configurations please check CONFIG file.")
parser.add_argument("-d", "--dataset", help="Choose the data source.", choices=["CIFAR10", "MNIST"], default="CIFAR10")
parser.add_argument("-i", "--index", help="Choose a specific image to reconstruct.", type=int, default=-1)
parser.add_argument("-b", "--batchsize", default=1, help="Mini-batch size", type=int)
parser.add_argument("-p", "--parameters", help="Load pre-trained model.", default=None)
parser.add_argument("-m", "--model", help="Network architecture.", choices=["CNN6", "CNN6-d", "FCN3"], default='CNN6')
args = parser.parse_args()
setup = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'dtype': torch.float32}


class DualStreamHandler:
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
log_file_path = 'Plots/output.txt'
dual_handler = DualStreamHandler(log_file_path)
sys.stdout = dual_handler

print(f'Running on {setup["device"]}, PyTorch version {torch.__version__}')
start_time = time.time()

def save_single_image(original, reconstructed):
    tp = torchvision.transforms.ToPILImage()

    plt.figure('origin')
    plt.imshow(original)
    plt.axis('off')
    plt.savefig(os.path.join(config['path_to_demo'], f'origin{args.index}.png'))

    reconstructed = torch.tensor(reconstructed) if isinstance(reconstructed, np.ndarray) else reconstructed
    reconstructed = reconstructed.squeeze()

    # Reshape based on dataset
    if args.dataset == "MNIST":
        reconstructed = reconstructed.view(1, 28, 28)  # MNIST is grayscale
    elif args.dataset == "CIFAR10":
        reconstructed = reconstructed.view(3, 32, 32)  # CIFAR10 is RGB

    # Check shape after reshaping
    print(f"Saving image with shape {reconstructed.shape}")

    plt.figure('reconstructed')
    plt.imshow(tp(reconstructed))
    plt.axis('off')
    plt.savefig(os.path.join(config['path_to_demo'], f'reconstructed_{args.index}.png'))

    plt.figure('rescale reconstructed')
    rescaled = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    plt.imshow(tp(rescaled))
    plt.axis('off')
    plt.savefig(os.path.join(config['path_to_demo'], f'rescale_reconstructed_{args.index}.png'))

def save_batch_images(original, reconstructed, index):
    tp = torchvision.transforms.ToPILImage()
    output_dir = os.path.join(config['path_to_demo'], f'batch_{index}')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure('origin')
    plt.imshow(original)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'origin.png'))

    reconstructed = torch.tensor(reconstructed) if isinstance(reconstructed, np.ndarray) else reconstructed
    reconstructed = reconstructed.squeeze()

    # Reshape based on dataset
    if args.dataset == "MNIST":
        reconstructed = reconstructed.view(1, 28, 28)  # MNIST is grayscale
    elif args.dataset == "CIFAR10":
        reconstructed = reconstructed.view(3, 32, 32)  # CIFAR10 is RGB

    # Check shape after reshaping
    print(f"Saving image {index} with shape {reconstructed.shape}")

    plt.figure('reconstructed')
    plt.imshow(tp(reconstructed))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'reconstructed.png'))

    plt.figure('rescale reconstructed')
    rescaled = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    plt.imshow(tp(rescaled))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'rescale_reconstructed.png'))

def processing_images(image, label, index, dataset, model_name, parameters):
    setup_cpu = {'device': torch.device('cpu'), 'dtype': torch.float32}

    # Load the model inside each process
    if model_name == 'CNN6':
        net = CNN6().to(**setup_cpu).eval()
    elif model_name == 'CNN6-d':
        net = CNN6d().to(**setup_cpu).eval()
    elif model_name == 'FCN3':
        net = FCN3().to(**setup_cpu).eval()

    if parameters:
        checkpoint = torch.load(parameters, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])

    pred_loss_fn = logistic_loss

    tt = torchvision.transforms.ToTensor()
    x = tt(image).unsqueeze(0).to(**setup_cpu)

    pred, x_shape = net(x)
    y = torch.tensor([0 if p > 0 else 1 for p in pred]).to(**setup_cpu)
    pred_loss = pred_loss_fn(inputs=pred, target=y)
    dy_dx = torch.autograd.grad(pred_loss, list(net.parameters()))
    original_dy_dx = [g.detach().clone() for g in dy_dx]
    original_dy_dx.reverse()
    modules = net.body[-1::-1]
    x_shape.reverse()
    k = None
    last_weight = []
    x_ = None

    for i in range(len(modules)):
        g = original_dy_dx[i].cpu().numpy()
        w = list(modules[i].layer.parameters())[0].detach().cpu().numpy()

        if k is None:
            udldu = np.dot(g.reshape(-1), w.reshape(-1))
            u = inverse_udldu(udldu, image_index=index)
            y = y.cpu().numpy()
            y = np.array([-1 if n == 0 else n for n in y], dtype=np.float32).reshape(-1, 1)
            y = y.mean() if y.mean() != 0 else 0.1
            k = -y / (1 + np.exp(u))
            k = k.reshape(1, -1).astype(np.float32)
        else:
            if isinstance(modules[i].act, nn.LeakyReLU):
                da = derive_leakyrelu(x_, slope=modules[i].act.negative_slope)
            elif isinstance(modules[i].act, nn.Identity):
                da = derive_identity(x_)

            out = x_
            if isinstance(modules[i].act, nn.LeakyReLU):
                out = inverse_leakyrelu(x_, slope=modules[i].act.negative_slope)
            elif isinstance(modules[i].act, nn.Identity):
                out = inverse_identity(x_)

            padding = modules[i - 1].layer.padding[0] if hasattr(modules[i - 1].layer, 'padding') else 0
            in_shape = np.array(x_shape[i - 1])
            in_shape[0] = 1
            x_mask = peeling(in_shape=in_shape, padding=padding)
            k = np.multiply(np.matmul(last_weight.transpose(), k)[x_mask], da.transpose())

        if isinstance(modules[i].layer, nn.Conv2d):
            x_, last_weight = r_gap(out=out, k=k, x_shape=x_shape[i], module=modules[i], g=g, weight=w)
        else:
            x_, last_weight = fcn_reconstruction(k=k, gradient=g), w

    if args.batchsize > 1:
        save_batch_images(image, x_, index)
    else:
        save_single_image(image, x_)
def main():
    train_sample, _ = dataloader(dataset=args.dataset, mode="attack", index=args.index, batchsize=args.batchsize,
                                 config=config)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.model == 'CNN6':
        net = CNN6().to(**setup).eval()
    elif args.model == 'CNN6-d':
        net = CNN6d().to(**setup).eval()
    else:
        net = FCN3().to(**setup).eval()

    if args.parameters:
        checkpoint = torch.load(args.parameters)
        ep = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f'Load model trained with {ep} epochs.')

    if args.batchsize > 1:
        with Pool(processes=args.batchsize) as pool:
            pool.starmap(processing_images, [
                (image, label, idx, args.dataset, args.model, args.parameters)
                for idx, (image, label) in enumerate(train_sample)
            ])

    else:
        image, label = train_sample
        processing_images(image, label, 0, args.dataset, args.model, args.parameters)

    print("Reconstruction completed.")


if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
sys.stdout = sys.__stdout__
dual_handler.file.close()

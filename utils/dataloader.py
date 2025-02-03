import torch
import torchvision
import random
import torchvision.transforms as transformers


def dataloader(dataset, mode, index, config):
    '''
    :param dataset: MNIST or CIFAR10 or CelebA
    :param mode: Train or reconstruction.
    :param index: Pick up a specific image.
    :param config: Some meta arguments.
    :return: Dataloader of pytorch in train mode and PIL image, as well as label in attack mode.
    '''
    path = config['path_to_dataset']
    if dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True)
    elif dataset.lower() == "mnist":
        trainset = torchvision.datasets.MNIST(root=path, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=path, train=False, download=True)
    elif dataset.lower() == "celeba":
        trainset = torchvision.datasets.CelebA(root=path, split='train', download=True)
        testset = torchvision.datasets.CelebA(root=path, split='test', download=True)
    else:
        raise ValueError("Unknown dataset.")

    if mode == "attack":

        if index == -1:
            trainloader = random.choice(list(trainset))
            testloader = random.choice(list(testset))
        else:
            trainloader = trainset[index]
            testloader = trainset[index]
        return trainloader, testloader

    elif mode == "train":
        channels = 1 if "mnist" in dataset.lower() else 3
        # no augmentation in this version
        trainset.transform = preprocessing(channels)
        testset.transform = preprocessing(channels)
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                                  num_workers=config["multithread"])
        testloader = torch.utils.data.DataLoader(testset, shuffle=True,
                                                 num_workers=config["multithread"])
        return trainloader, testloader
    else:
        raise ValueError("Unknown mode.")


def preprocessing(channel):
    transform = transformers.Compose([
        transformers.ToTensor(),
        transformers.Normalize([0.5]*channel, [0.5]*channel)
    ])
    return transform


import cifar10
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)




trainset = torchvision.datasets.CIFAR10(root='./data1', train=True,download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data1', train=False,download=False, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

#随机获取部分数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#显示数据
imshow(torchvision.utils.make_grid(images))

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
print(" ".join('%5s' % classes[labels[j]] for j in range(4)))
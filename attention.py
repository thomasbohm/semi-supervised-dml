from PIL import Image
from net.gnn import GNNModel
from torchvision import transforms
from net.load_net import load_resnet50
import torch

img = Image.open('./data/CARS/images/001/000018.jpg')

mean = [0.485, 0.456, 0.406]
std = [0.299, 0.224, 0.225]
sz_resize = 256
sz_crop = 227
normalize_transform = transforms.Normalize(mean=mean, std=std)

transform_train = [
    transforms.RandomResizedCrop(sz_crop),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_transform
]
transform_train = transforms.Compose(transform_train)

img = transform_train(img)
img = img[None,:,:,:]

print('Loaded img with shape', img.shape)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


resnet, embed_dim = load_resnet50(
    num_classes=98,
    pretrained_path='./results/CARS/2022-11-19_02:19:16/CARS_10.0_best.pth',
    reduction=4,
    neck=False
)
resnet.to(device)
print('Loaded ResNet.')



gnn = GNNModel(
    device,
    embed_dim = embed_dim,
    output_dim = 98,
    num_layers = 1,
    num_heads = 6,
    num_proxies = 98,
    add_mlp = False,
    gnn_conv = 'GAT',
    gnn_fc = False,
    reduction_layer = False
)
gnn = gnn.to(device)
gnn.load_state_dict(torch.load('./results/CARS/2022-11-19_02:19:16/CARS_10.0_gnn_best.pth'))
print('Loaded GNN.')
print(gnn)

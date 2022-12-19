from PIL import Image
from dataset.m_per_class_sampler import MPerClassSampler
from dataset.ssl_dataset import create_datasets, get_transforms
from net.gnn import GNNModel
from torchvision import transforms
from net.load_net import load_resnet50
import torch
import os.path as osp
from torch.utils.data import DataLoader

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


#embeds, preds = resnet(img)
#print('ResNet prediction with shape', preds.shape, ':', torch.argmax(preds))

#embeds_gnn, preds_gnn = gnn(embeds,
#    proxy_idx=None,
#    kclosest=12,
#    true_proxies=torch.tensor([0])
#)
#print('GNN embeds:', embeds_gnn.shape, 'GNN preds:', preds_gnn.shape)
#print(torch.argmax(preds_gnn))









trans_train, trans_train_strong, trans_eval = get_transforms(
    'randaugment',
    False,
    4,
    5
)
print('Transform (train_weak, train_strong, eval):\n{}\n{}\n{}'.format(
    trans_train,
    trans_train_strong,
    trans_eval
))
dset_lb, dset_ulb, dset_eval = create_datasets(
    osp.join('data/CARS', 'images'),
    98,
    1.0,
    trans_train,
    trans_train_strong,
    trans_eval,
    False
)

class_per_batch = 12
elements_per_class = 5

batch_size_lb = class_per_batch * elements_per_class
batch_size_ulb = 7 * batch_size_lb

num_batches = max(len(dset_ulb) // batch_size_ulb, len(dset_lb) // batch_size_lb)

sampler_lb = MPerClassSampler(
    dset_lb.targets,
    m=elements_per_class,
    batch_size=batch_size_lb,
    length_before_new_iter=batch_size_lb * num_batches
)
dl_train_lb = DataLoader(
    dset_lb,
    batch_size=batch_size_lb,
    sampler=sampler_lb,
    drop_last=True,
    pin_memory=True,
)


x, y = next(dl_train_lb)
print('Loaded batch:')
print(x.shape)

embeds, preds = resnet(x)

embeds_gnn, preds_gnn = gnn(embeds,
    proxy_idx=None,
    kclosest=12,
    true_proxies=y
)

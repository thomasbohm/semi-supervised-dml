from PIL import Image
from torchvision import transforms

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

print(img.shape)
print(img[None,:,:,:].shape)
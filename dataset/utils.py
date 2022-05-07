from collections import defaultdict
import math
import random
import PIL
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def get_list_of_inds(dataset):
    ddict = defaultdict(list)
    for idx, label in enumerate(dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])
    return list_of_indices_for_each_class


def GL_orig_RE(sz_crop=[384, 128],
               mean=[0.485, 0.456, 0.406],
               std=[0.299, 0.224, 0.225],
               is_train=True,
               RE=False):
    
    sz_resize = 256
    sz_crop = 227
     
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    
    if is_train and RE:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5,
                          mean=(0.4914, 0.4822, 0.4465))
        ])
    elif is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
            transforms.ToTensor(),
            normalize_transform
        ])
    
    print(transform)
    return transform


class RandomErasing(object):
    """
    From https://github.com/zhunzhong07/Random-Erasing
    Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


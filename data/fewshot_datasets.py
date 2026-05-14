import json
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        #print("image path:",image_path)
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397',
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir_rel_to_root", "json_rel_to_root"]
    "flower102": ["oxford_flowers/jpg", "oxford_flowers/split_zhou_OxfordFlowers.json"],
    "food101": ["food-101/images", "food-101/split_zhou_Food101.json"],
    "dtd": ["dtd/images", "dtd/split_zhou_DescribableTextures.json"],
    "pets": ["oxford_pets/images", "oxford_pets/split_zhou_OxfordPets.json"],
    "sun397": ["sun397/SUN397", "sun397/split_zhou_SUN397.json"],
    "caltech101": ["caltech-101/101_ObjectCategories", "caltech-101/split_zhou_Caltech101.json"],
    "ucf101": ["ucf101/UCF-101-midframes", "ucf101/split_zhou_UCF101.json"],
    "cars": ["stanford_cars", "stanford_cars/split_zhou_StanfordCars.json"],
    "eurosat": ["eurosat/2750", "eurosat/split_zhou_EuroSAT.json"],
}


def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    if set_id.lower() == 'aircraft':
        aircraft_root = os.path.join(root, "fgvc_aircraft")
        return Aircraft(aircraft_root, mode, n_shot, transform)

    img_rel, json_rel = path_dict[set_id.lower()]
    image_path = os.path.join(root, img_rel)
    json_path = os.path.join(root, json_rel)
    
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

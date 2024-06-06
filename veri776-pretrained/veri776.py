from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import torch


import os


def parse_xml(xml_path):
    with open(xml_path) as f:
        et = ET.fromstring(f.read())

        image_paths = []
        vehicle_ids = []
        class_map = dict()
        cur_class = 0


        for item in et.iter('Item'):
            image_paths.append(item.attrib['imageName'])

            vehicle_id = int(item.attrib['vehicleID'])
            vehicle_ids.append(vehicle_id)

            if vehicle_id not in class_map:
                class_map[vehicle_id] = cur_class
                cur_class += 1

        return image_paths, vehicle_ids, class_map
    


def get_veri776_train(veri776_path, num_workers, batch_size, transform, drop_last=False, shuffle=False):
    
    img_paths, vehicle_ids, class_map = parse_xml(os.path.join(veri776_path, 'train_label.xml'))
    img_paths = [os.path.join(veri776_path, 'image_train', path) for path in img_paths]
    train_set = Veri776Train(img_paths, vehicle_ids, class_map, transform)


    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
   

def get_veri776_test(veri_776_path, transform):
    img_file_names, vehicle_ids, _ = parse_xml(os.path.join(veri_776_path, 'test_label.xml'))

    return Veri776Test(img_file_names, vehicle_ids, transform, veri_776_path)


class Veri776Test:
    def __init__(self, img_file_names, vehicle_ids, transform, veri776_root):
        self.img_file_names = np.array(img_file_names) # for indexing in __getitem__
        self.vehicle_ids = vehicle_ids
        self.transform = transform
        self.veri776_root = veri776_root


    def __len__(self):
        return len(self.img_file_names)


    def __iter__(self):
        for i in range(len(self)):
            img = Image.open(os.path.join(self.veri776_root, 'image_test', self.img_file_names[i]))

            if self.transform is not None:
                img = self.transform(img)


            yield self.img_file_names[i], img


class Veri776Train(Dataset):
    def __init__(self, img_paths, vehicle_ids, class_map, transform):
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map # map the vehicle id to the label used for classification
        self.transform = transform

        self.class_tree = self.build_class_tree(vehicle_ids, class_map, img_paths) # maps the class to a list of images which has that class

    def build_class_tree(self, vehicle_ids, class_map, img_paths):
        class_tree = defaultdict(list)
        for id, path in zip(vehicle_ids, img_paths):
            class_tree[class_map[id]].append(path) 

        return class_tree
        

    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        anchor_img = Image.open(self.img_paths[index])

        label = self.class_map[self.vehicle_ids[index]]
        positive_img_path = np.random.choice(self.class_tree[label])
        positive_img = Image.open(positive_img_path)


        negative_img_class = self.random_number_except(0, len(self.class_map), label)
        negative_img_path = np.random.choice(self.class_tree[negative_img_class])
        negative_img = Image.open(negative_img_path)

        if self.transform is not None:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            anchor_img = self.transform(anchor_img)


        return torch.stack((anchor_img, positive_img, negative_img), dim=0), torch.tensor([label, label, negative_img_class])
    

    def random_number_except(self, range_start, range_end, excluded_number):
        numbers = list(range(range_start, range_end))  # Create a list of numbers in the specified range
        numbers.remove(excluded_number)  # Remove the excluded number from the list
        return np.random.choice(numbers)
    


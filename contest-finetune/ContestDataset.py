from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import torch
from enum import Enum

import os


def parse_xml(xml_path):
    with open(xml_path) as f:
        et = ET.fromstring(f.read())

        image_paths = []
        vehicle_ids = []
        class_map = dict()
        cur_class = 0

        id_count = defaultdict(int)


        for item in et.iter('Item'):
            id_count[int(item.attrib['vehicleID'])] += 1


        for item in et.iter('Item'):
            id = int(item.attrib['vehicleID'])
            if id_count[id] > 1:
                image_paths.append(item.attrib['imageName'])
                vehicle_ids.append(id)

                if id not in class_map:
                    class_map[id] = cur_class
                    cur_class += 1

        return image_paths, np.array(vehicle_ids), class_map


def get_contest_test(contest_path, transform, queries=None):
    xml_path = 'test_label.xml'
    img_file_names, vehicle_ids, _ = parse_xml(os.path.join(contest_path, xml_path))

    return ContestTest(img_file_names, vehicle_ids, transform, contest_path, 'image_test', queries)



def get_contest_val(contest_path, transform):
    xml_path = 'val_label.xml'

    img_file_names, vehicle_ids, _ = parse_xml(os.path.join(contest_path, xml_path))

    return ContestTest(img_file_names, vehicle_ids, transform, contest_path, 'image_val')


def get_contest_train(contest_path, num_workers, batch_size, transform, disable_join):
    '''
        contest_path (str): the path to contest dataset root 
    '''
    
    xml_path = 'joined_train_label.xml' if not disable_join else 'train_label.xml'
    img_paths, vehicle_ids, class_map = parse_xml(os.path.join(contest_path, xml_path))

    image_dir = 'joined_image_train' if not disable_join else 'image_train'
    img_paths = [os.path.join(contest_path, image_dir, path) for path in img_paths]
    train_set = ContestTrain(img_paths, vehicle_ids, class_map, transform)

    print(f'using {xml_path} and {image_dir}')
    print(f'Found {len(class_map)} identities')

    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size), len(class_map)


def get_contest_final(contest_path, num_workers, batch_size, transform, disable_join):
    '''
        contest_path (str): the path to contest dataset root 
    '''
    
    xml_path = 'final.xml' if not disable_join else 'train_label.xml'
    img_paths, vehicle_ids, class_map = parse_xml(os.path.join(contest_path, xml_path))

    image_dir = 'final' if not disable_join else 'image_train'
    img_paths = [os.path.join(contest_path, image_dir, path) for path in img_paths]
    train_set = ContestTrain(img_paths, vehicle_ids, class_map, transform)

    print(f'using {xml_path} and {image_dir}')
    print(f'Found {len(class_map)} identities')

    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size), len(class_map)


class ContestTrain(Dataset):
    def __init__(self, img_paths, vehicle_ids, class_map, transform, crop_threshold=128):

        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map # map the vehicle id to the label used for classification
        self.transform = transform

        self.class_tree = self.build_class_tree(vehicle_ids, class_map, img_paths) # maps the class to a list of images which has that class
        self.crop_threshold = crop_threshold


    def build_class_tree(self, vehicle_ids, class_map, img_paths):
        class_tree = defaultdict(list)
        for id, path in zip(vehicle_ids, img_paths):
            class_tree[class_map[id]].append(path) 

        return class_tree
        

    def __len__(self):
        return len(self.img_paths)
    

    def random_crop(self, img, proba):
        if np.random.random() > proba:
            return img
        
        w, h = img.size

        rand = np.random.random()
        if rand < 0.2: # left
            return img.crop((0, 0, w / 2, h))
        if rand < 0.4: # top
            return img.crop((0, 0, w, h / 2))
        if rand < 0.6: # right
            return img.crop((w / 2, 0, w, h))
        if rand < 0.8: # top-right
            return img.crop((w / 2, 0, w, h / 2))
        return img.crop((0, 0, w / 2, h / 2)) # top left


    def __getitem__(self, index):
        anchor_img = Image.open(self.img_paths[index])
        label = self.class_map[self.vehicle_ids[index]]

        anchor_width, anchor_height = anchor_img.size
        if anchor_height >= self.crop_threshold and anchor_width >= self.crop_threshold and np.random.random() < 0.1:
            positive_img = self.random_crop(anchor_img, proba=0.7)
            anchor_img = self.random_crop(anchor_img, proba=0.7)
        else:
            positive_img_path = np.random.choice(self.class_tree[label])
            positive_img = Image.open(positive_img_path)


        negative_img_class = self.random_number_except(0, len(self.class_tree), label)
        negative_img_path = np.random.choice(self.class_tree[negative_img_class])
        negative_img = Image.open(negative_img_path)

        negative_width, negative_height = negative_img.size
        if negative_height >= self.crop_threshold and negative_width >= self.crop_threshold:
            negative_img = self.random_crop(negative_img, proba=0.1)                 


        if self.transform is not None:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            anchor_img = self.transform(anchor_img)


        return torch.stack((anchor_img, positive_img, negative_img), dim=0), torch.tensor([label, label, negative_img_class])
    

    def random_number_except(self, range_start, range_end, excluded_number):
        numbers = np.arange(range_start, range_end)  # Create a list of numbers in the specified range
        numbers = np.delete(numbers, excluded_number)  # Remove the excluded number from the list
        return np.random.choice(numbers)


class ContestTest:
    def __init__(self, img_file_names, vehicle_ids, transform, dataset_root, image_path, query_indices=None):
        self.img_file_names = np.array(img_file_names) # for indexing in __getitem__
        self.vehicle_ids = np.array(vehicle_ids)
        self.transform = transform

        self.image_path = image_path
        self.root = os.path.join(dataset_root, image_path)


        self.queries, self.gt = self.deterministic_partition(query_indices) if query_indices is not None else self.partition()


    def deterministic_partition(self, query_indices):
        gt_index = np.arange(0, len(self.vehicle_ids))
        
        gt_indices = [idx for idx in gt_index if idx not in query_indices]


        queries = [(os.path.join(self.root, self.img_file_names[query_idx]), self.vehicle_ids[query_idx]) for query_idx in query_indices]
        gt = [(os.path.join(self.root, self.img_file_names[gt_idx]), self.vehicle_ids[gt_idx]) for gt_idx in gt_indices]

        return queries, gt
    

    def partition(self):

        queries = []
        gt = []

        start = 0
        for idx in range(len(self.vehicle_ids)):
            if idx == len(self.vehicle_ids) - 1 or self.vehicle_ids[idx + 1] != self.vehicle_ids[idx]:
                query_idx = np.random.randint(start, idx + 1)

                gt_index = np.arange(start, idx + 1)
                gt_indices = gt_index[gt_index != query_idx]

                start = idx + 1

                gt += [(os.path.join(self.root, self.img_file_names[gt_idx]), self.vehicle_ids[gt_idx]) for gt_idx in gt_indices]

                queries.append((os.path.join(self.root, self.img_file_names[query_idx]), self.vehicle_ids[query_idx]))

        return queries, gt


    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.image_path, self.img_file_names[idx]))

        if self.transform is not None:
            img = self.transform(img)

        return img
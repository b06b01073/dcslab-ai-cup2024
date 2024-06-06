from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from model import Resnet101IbnA
from argparse import ArgumentParser
import os
from Transforms import get_test_transform
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np


def get_class_imgs(xml_path, cls):

    img_paths = []
    xml_prefix, _ = os.path.split(xml_path)
    with open(xml_path) as f:
        et = ET.fromstring(f.read())

        for item in et.iter('Item'):
            if int(item.attrib['vehicleID']) == cls:
                img_paths.append(os.path.join(xml_prefix, 'image_test', item.attrib['imageName']))

    return img_paths



def visualize(X, y, out):
    '''
        X: a list of embedding features
        y: the corresponding label
    '''
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(X)

    # Visualize the result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, marker='o', s=30)
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(out)

    plt.show()


@torch.no_grad()
def get_data(model_params, vehicle_ids, xml_path):
    imgs_ls = [(id, get_class_imgs(xml_path, id)) for id in vehicle_ids] # not the most efficient algo, but anyway
    test_transform = get_test_transform()

    net = Resnet101IbnA()
    net = torch.load(model_params)
    net = net.to('cpu')
    net.eval()

    X = []
    y = []


    for id, cls_imgs in imgs_ls:
        for img in tqdm(cls_imgs, dynamic_ncols=True, desc=f'id: {id}'):
            input = Image.open(img)
            input = test_transform(input)
            eu_feat, cos_feat, _ = net(input.unsqueeze(dim=0))
            
            y.append(id)
            X.append(eu_feat.squeeze().numpy())

    return X, y


def run_tsne(model_params, vehicle_ids, xml_path, out):
    X, y = get_data(model_params, vehicle_ids, xml_path)    

    
    visualize(np.array(X), np.array(y), out)





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', '-p', type=str, default='run3/9.pth')
    parser.add_argument('--ids', nargs='+', default=[2, 5, 9, 14, 546, 653, 768, 38, 42, 402, 421, 281, 776, 150])
    parser.add_argument('--xml_path', '-x', type=str, default='../veri776/test_label.xml')
    parser.add_argument('--out', type=str)


    args = parser.parse_args()

    ids = [int(id) for id in args.ids]

    run_tsne(args.params, ids, args.xml_path, args.out)
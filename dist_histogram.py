import numpy as np
from argparse import ArgumentParser
from torchvision.io import read_image
from torchvision import transforms
from tqdm import tqdm
import torch
import os
import random
import matplotlib.pyplot as plt


date_list = ['0902_150000_151900', '0902_190000_191900', '0903_150000_151900', '0903_190000_191900',
            '0924_150000_151900', '0924_190000_191900', '0925_150000_151900', '0925_190000_191900',
            '1015_150000_151900', '1015_190000_191900', '1016_150000_151900', '1016_190000_191900']

WIDTH = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'model is running on {device}.')

def draw_histogram(data, name):
    """
    Draws a histogram of the given data.

    Args:
    - data (list or np.array): The data to plot in the histogram.
    - name (str): The filename to save the histogram as.
    """
    hist, bins = np.histogram(data, bins=np.arange(-0.5, 1, 0.05))
    plt.bar(bins[:-1], hist, width=0.05, color='skyblue', edgecolor='black')
    plt.xlabel('dist')
    plt.ylabel('num')

    plt.savefig(f'{name}.jpg')
    plt.close()
def draw_histogram2(data1, data2, name):
    """
    Draws a comparative histogram of two sets of data.

    Args:
    - data1 (list or np.array): The first set of data to plot.
    - data2 (list or np.array): The second set of data to plot.
    - name (str): The filename to save the histogram as.
    """
    hist1, bins1 = np.histogram(data1, bins=np.arange(-0.5, 1, 0.05))
    hist2, bins2 = np.histogram(data2, bins=np.arange(-0.5, 1, 0.05))
    plt.bar(bins1[:-1], hist1, width=0.05, color='skyblue', edgecolor='black', alpha=0.5, label='intra')
    plt.bar(bins2[:-1], hist2, width=0.05, color='salmon', edgecolor='black', alpha=0.5, label='inter')
    plt.xlabel('dist')
    plt.ylabel('num')

    plt.savefig(f'{name}.jpg')
    plt.close()

def get_random_id(current_id, id_list):
    """
    Get a random ID from id_list that is different from current_id.

    Args:
    - current_id (str): The current ID to avoid.
    - id_list (list): The list of all IDs to choose from.

    Returns:
    - random_id (str): A random ID different from current_id.
    """
    random_id = random.choice(id_list)
    while(random_id == current_id):
        random_id = random.choice(id_list)
        
    return random_id

def cal_ave_embedding(embedding_dict):
    """
    Calculate the average embedding for each camera in embedding_dict.

    Args:
    - embedding_dict (dict): Dictionary with camera IDs as keys and list of embeddings as values.

    Returns:
    - dist_list (list): List of average embeddings.
    - all_keys (list): List of camera IDs corresponding to the embeddings.
    """
    all_keys = list(embedding_dict.keys())
    dist_list = []
    for key in all_keys:
        dist_list.append(sum(embedding_dict[key]) / len(embedding_dict[key]))

    return dist_list, all_keys

@torch.no_grad()
def get_emdebbing(date_folder, id_, transform, extractor):
    """
    Get embeddings for all images of a given ID on a specific date.

    Args:
    - date_folder (str): Path to the folder containing the images for the date.
    - id_ (str): ID of the objects to process.
    - transform (transforms.Compose): Transformation to apply to the images.
    - extractor (torch model): Pre-trained model to extract embeddings.

    Returns:
    - embedding_list (list): List of average embeddings for each camera.
    - cam_list (list): List of camera IDs corresponding to the embeddings.
    """
    id_folder = os.path.join(date_folder, id_)
    object_list = os.listdir(id_folder)

    cam_set = set()
    embedding_dict = {}

    for object_ in object_list:
        # read img
        img_path = os.path.join(id_folder, object_)
        img = transform(read_image(img_path)/255).to(device)

        # get embedding of current object
        _, embedding, _ = extractor(torch.unsqueeze(img,0))
        # print(f'shape of embedding : {embedding.shape}')

        cam = object_.split('_')[0]

        if cam not in cam_set:
            cam_set.add(cam)
            embedding_dict[cam] = [embedding]
        else:
            embedding_dict[cam].append(embedding)

    embedding_list, cam_list = cal_ave_embedding(embedding_dict)
    return embedding_list, cam_list



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet101_ibn_a', help='the name of the pre-trained PyTorch model')
    parser.add_argument('--date', '-d', type=str, default='0902_150000_151900')
    parser.add_argument('--draw', type=bool, default=False)
    args = parser.parse_args()
    


    # Load the pre-trained model for feature extraction
    extractor = torch.hub.load('b06b01073/dcslab-ai-cup2024', args.model) # 將 fine_tuned 設為 True 會 load fine-tuned 後的 model

    extractor = extractor.to(device)
    extractor.eval()
    
    # Resize and Normalize image pixels before feeding them to the model
    transform = transforms.Compose([
        transforms.Resize((WIDTH, WIDTH)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    intra_dist = []
    inter_dist = []


    if args.draw != True:
        date_folder = os.path.join(f'cross_cams/{args.date}')
        id_list = os.listdir(date_folder)

        for id_ in tqdm(id_list, desc=f'{args.date}', dynamic_ncols=True):
            
            current_embedding_list, current_cam_list = get_emdebbing(date_folder, id_, transform, extractor)
            
            for current_embedding, cam in zip(current_embedding_list, current_cam_list):
                while(1):
                    random_id = get_random_id(id_, id_list)

                    # verify if it's the same camera.
                    inter_obj_list = os.listdir(os.path.join(date_folder, random_id))
                    inter_obj_cam = inter_obj_list[0].split('_')[0]
                    
                    if inter_obj_cam != cam:
                        break
            
                diff_embedding_list, diff_cam_list = get_emdebbing(date_folder, random_id, transform, extractor)
                
                # print(f'current_embedding_list : {current_embedding_list}')
                # print(f'current_cam_list : {current_cam_list}')
                # print(f'diff_embedding_list : {diff_embedding_list}')
                # print(f'diff_cam_list : {diff_cam_list}')   
                
                # calculate inter-class distance
                dist = torch.nn.functional.cosine_similarity(current_embedding, diff_embedding_list[0])
                inter_dist.append(dist.item())

            # calculate intra-class distance
            if len(current_embedding_list) > 1:
                # print(f'current_embedding_list : {current_embedding_list}')

                list_len = len(current_embedding_list)
                for i in range(list_len):
                    for j in range(i+1, list_len):
                        dist = torch.nn.functional.cosine_similarity(current_embedding_list[i], current_embedding_list[j])
                        intra_dist.append(dist.item())

                # print(f'inter_dist : {inter_dist}')
                # print(f'intra_dist : {intra_dist}')

        np.save(f'histogram/{args.date}_intra', intra_dist)
        np.save(f'histogram/{args.date}_inter', inter_dist)
    
    else:
        intra_list = []
        inter_list = []
        for date in date_list:
            intra = np.load(f'histogram/{date}_intra.npy')
            inter = np.load(f'histogram/{date}_inter.npy')
            intra_list.append(intra)
            inter_list.append(inter)
        intra_list = np.concatenate(intra_list, axis=0)
        inter_list = np.concatenate(inter_list, axis=0)
        
        draw_histogram(intra_list, 'intra_dist_hist_0.05')
        draw_histogram(inter_list, 'inter_dist_hist_0.05')
        draw_histogram2(intra_list, inter_list, 'both')

            
import torch
import os
import numpy as np
from argparse import ArgumentParser
from FrameLoader import FrameLoader
from torchvision import transforms
from Cropper import Cropper
from tqdm import tqdm
from Matcher import Matcher



def cluster_max(list_1, list_2):
    max_sim = -100
    for i in list_1:
        for j in list_2:
            similarity = torch.nn.functional.cosine_similarity(i, j, dim=0)
            if similarity > max_sim:
                max_sim = similarity

    return max_sim
    
def cluster_min(list_1, list_2):
    min_sim = 100
    for i in list_1:
        for j in list_2:
            similarity = torch.nn.functional.cosine_similarity(i, j, dim=0)
            if similarity < min_sim:
                min_sim = similarity
    return min_sim

def cluster_ave_v1(list_1, list_2):
    len_1 = len(list_1)
    len_2 = len(list_2)
    sum_sim = 0
    for i in list_1:
        for j in list_2:
            similarity = torch.nn.functional.cosine_similarity(i, j, dim=0)
            sum_sim += similarity / (len_1 * len_2)
    return sum_sim

def cluster_ave_v2(list_1, list_2):
    mean_sim_1 = torch.mean(list_1, dim=0)
    mean_sim_2 = torch.mean(list_2, dim=0)
    
    similarity = torch.nn.functional.cosine_similarity(mean_sim_1, mean_sim_2, dim=0)
           
    return similarity

MODE = {
    'max' : cluster_max,
    'min' : cluster_min,
    'ave_v1' : cluster_ave_v1,
    'ave_v2' : cluster_ave_v2
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--date', '-d', default='0902_150000_151900', type=str)
    parser.add_argument('--model', '-m', type=str, default='resnet101_ibn_a', help='the name of the pre-trained PyTorch model')
    parser.add_argument('--width', '-w', type=int, default=224)
    parser.add_argument('--threshold', '-t', type=float, default=0.4)
    parser.add_argument('--mode', type=str, default='max', help='Specify the distance calculation method to be used.')
    parser.add_argument('--finetune', default=False, type=bool, help='Specify whether in finetune mode')
    args = parser.parse_args()




    out_folder = os.path.join(f'final_result','labels', f'{args.model}_{args.mode}_{int(args.threshold)}',f'{args.date}')


    # To check if it's in fine-tune mode
    if args.finetune:
        args.threshold /= 100



    # Load the pre-trained model for feature extraction
    extracter = torch.hub.load('b06b01073/dcslab-ai-cup2024', args.model) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'model is running on {device}.')
    extracter = extracter.to(device)
    extracter.eval()

    # Normalize image pixels before feeding them to the model
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    

    matched_set = {}
    ID_ = -1 #cause id start with 0
    

    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    
    for cam in range(8):

        # Initialize Cropper and Matcher
        cropper = Cropper(args.width, cam, 0)
        
        #basic threshold = 0.5
        matcher = Matcher(threshold=args.threshold)

        #check if the output of the single camera tracking.
        if not os.path.exists(f'../../dcslab-ai-cup2024/aicup_gt/labels/{args.date}/{cam}'):
            print(f'The output from camera {cam} does not exist.')
            continue
        # Set up the FrameLoader to load frames
        frameloader = FrameLoader(f'../../IMAGE/{args.date}', f'../../dcslab-ai-cup2024/aicup_gt/labels/{args.date}/{cam}')

        # Load data for the current camera
        imgs, labels = frameloader.load(cam)

        with torch.no_grad():
            if len(matched_set) == 0:
                for i in tqdm(range(len(imgs)), dynamic_ncols=True, desc = f'{cam}/7'):
                    current_objects, info_list, info_list_norm = cropper.crop_frame(image_path=imgs[i], label_path=labels[i], multi=True)

                    # Extract features for each cropped object
                    for n in range(len(current_objects)):
                        img = transform(current_objects[n])                
                        _, feature, _ = extracter(torch.unsqueeze(img,0).to(device))
                        
                        id_ = int(info_list[n][4])

                        if id_ not in matched_set:
                            matched_set[id_] = [feature, [i], [info_list_norm[n][0:4]]]
                        else:
                            matched_set[id_][0] = torch.cat((matched_set[id_][0], feature), dim=0)
                            matched_set[id_][1].append(i)
                            matched_set[id_][2].append(info_list_norm[n][0:4])
                
                frame_wrote = set()
                for key, value in matched_set.items():
                    ID_ += 1
                    for i in range(len(value[0])):
                        frame_wrote.add(value[1][i]+1)
                        f = open(f'{out_folder}/{cam}_{value[1][i]+1:05}.txt', 'a')
                        f.write(f'0 {value[2][i][0]} {value[2][i][1]} {value[2][i][2]} {value[2][i][3]} {key}\n')
                        f.close()


                for i in range(1, 361):
                    if i not in frame_wrote:
                        f = open(f'{out_folder}/{cam}_{i:05}.txt', 'a')
                        f.write(f'')
                        f.close()

            else:
                current_set = {}

                for i in tqdm(range(len(imgs)), dynamic_ncols=True, desc = f'{cam}/7'):
                    current_objects, info_list, info_list_norm = cropper.crop_frame(image_path=imgs[i], label_path=labels[i], multi=True)

                    # Extract features for each cropped object
                    for n in range(len(current_objects)):
                        img = transform(current_objects[n])                
                        _, feature, _ = extracter(torch.unsqueeze(img,0).to(device))

                        
                        id_ = int(info_list[n][4])
                        if id_ not in current_set:
                            current_set[id_] = [feature, [i], [info_list_norm[n][0:4]]]
                        else:
                            current_set[id_][0] = torch.cat((current_set[id_][0], feature), dim=0)
                            current_set[id_][1].append(i)
                            current_set[id_][2].append(info_list_norm[n][0:4])


                # calculate Cosine similarity matrix
                cosine_sim_matrix = []
                for key_1, value_1 in matched_set.items():
                    similarity = []
                    for key_2, value_2 in current_set.items():
                        similarity.append(MODE[args.mode](value_1[0], value_2[0]))
                    cosine_sim_matrix.append(similarity)
                
                # match

                current_set, matched_ID = matcher.multi_match(torch.tensor(cosine_sim_matrix), matched_set, current_set)


                frame_wrote = set()
                tmp = current_set.copy()
                for key, value in current_set.items():
                    if key not in matched_ID:
                        ID_ += 1
                        tmp[ID_] = tmp.pop(key)
                    for i in range(len(value[0])):
                        frame_wrote.add(value[1][i]+1)
                        f = open(f'{out_folder}/{cam}_{value[1][i]+1:05}.txt', 'a')
                        if key in matched_ID:
                            f.write(f'0 {value[2][i][0]} {value[2][i][1]} {value[2][i][2]} {value[2][i][3]} {key}\n')
                        else:
                            f.write(f'0 {value[2][i][0]} {value[2][i][1]} {value[2][i][2]} {value[2][i][3]} {ID_}\n')

                        f.close()
                current_set = tmp.copy()


                for i in range(1, 361):
                    if i not in frame_wrote:
                        f = open(f'{out_folder}/{cam}_{i:05}.txt', 'a')
                        f.write(f'')
                        f.close()

                
                # remove unmatched car
                tmp = {}
                for key in matched_ID:
                    tmp[key] = matched_set[key]
                matched_set = tmp.copy()
                
                for key, value in current_set.items():


                    if key not in matched_set:
                        matched_set[key] = [value[0], value[1], value[2]]
                    else:
                        matched_set[key][0] = torch.cat((matched_set[key][0], value[0]), dim=0)
                        for i in range(len(value[0])): 
                            matched_set[key][1].append(value[1][i])
                            matched_set[key][2].append(value[2][i])

                    
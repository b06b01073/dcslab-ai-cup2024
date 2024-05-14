import torch
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from FrameLoader import FrameLoader
from Cropper import Cropper
from Matcher import Matcher
from torchvision import transforms
from tqdm import tqdm
import json

# Define a class to manage color palettes for different IDs
class Palette:
    def __init__(self):     
        self.colors = {}
        
    def get_color(self, id):
        # Generate a random color for a new ID or return an existing color for a known ID
        if not id in self.colors:
            color = list(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))

            self.colors[id] = color

        return self.colors[id]





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--frame_dir', '-f', type=str, help='Directory containing input video frames.')
    parser.add_argument('--label_dir', '-l', type=str, help='Directory containing labels for input frames.')
    parser.add_argument('--out', type=str, help='Directory to save the output video.')
    parser.add_argument('--width', '-w', type=int, default=224)
    parser.add_argument('--buffer_size', type=int, default=1, help='size limit of the object buffer.')
    parser.add_argument('--visualize', '-v', type=str, default=False, help='Set to "True" to enable visualization of tracking results.')
    parser.add_argument('--threshold', type=float, help='Set the threshold for tracking objects.')
    parser.add_argument('--lambda_value', type=float, help='Set the lambda value for re-ranking.')
    parser.add_argument('--re_rank', type=bool, default=False)
    args = parser.parse_args()


    # Create output directory if it does not exist
    if not os.path.exists(f'{args.out}'):
        os.mkdir(args.out)

    # Set up the FrameLoader to load frames
    frameloader = FrameLoader(args.frame_dir, args.label_dir)


    # Normalize image pixels before feeding them to the model
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Relative paths to folders containing files generated from individual models
    models_files_folder = [
        'result_label/resnet101_ibn_a',
        'result_label/resnext101_ibn_a',
        'result_label/densenet169_ibn_a',
        'result_label/se_resnet101_ibn_a',
        'result_label/swin_reid',
    ]
    
    
    
    # Iterate over each camera
    for cam in range(8):
        
        # Initialize frame ID for writing to output file
        frame_id = 1

        # Load data for the current camera
        imgs, labels = frameloader.load(cam)

        # Create video writer if visualization is enabled
        if args.visualize:
            video_dir = os.path.join(f'video_result_ensemble')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            save_dir = os.path.join(video_dir, f'{args.out.split("/")[1]}')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            

            save_path = os.path.join(save_dir, f'{cam}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_out = cv2.VideoWriter(save_path, fourcc, 2, (1280,  720)) 

        # Initialize Cropper and Matcher
        cropper = Cropper(args.width)

        #basic threshold = 0.5
        matcher = Matcher(threshold=args.threshold, buffer_size=args.buffer_size, lambda_value=args.lambda_value)
        palette = Palette()

        # Perform object tracking for each frame
        with torch.no_grad():
            for i in tqdm(range(len(imgs))):

                #load info list for this frame
                with open(os.path.join(models_files_folder[0],str(cam),f'{frame_id}_info.json'),'r') as f:
                    info_list = json.load(f)

                #load normalized info list for this frame
                with open(os.path.join(models_files_folder[0],str(cam),f'{frame_id}_info_norm.json'),'r') as f:
                    info_list_norm = json.load(f)

                #load embeddings for this frame
                object_embeddings = np.load(os.path.join(models_files_folder[0],str(cam),f'{frame_id}_embeddings.npy'))


                # Open a text file to record the label of each frame
                out = os.path.join(args.out, f'{cam}')
                if not os.path.exists(out):
                    os.mkdir(out)
                f = open(f'{out}/{cam}_{frame_id:05}.txt', 'w')



                #load distance matrices from all models
                model_dist_mats = []
                for model in models_files_folder:
                    matrix = torch.load(os.path.join(model,str(cam), f'{frame_id}.pt'))
                    model_dist_mats.append(matrix)

 

                # Match object embeddings to previous frames
                id_list=  matcher.get_ensemble_id_list(np.array(object_embeddings), info_list, model_dist_mats, args.re_rank)

                # Record coordinates and IDs to the output file
                for n in range(len(info_list)):
                    f.write(f'{cam} {info_list_norm[n][0]} {info_list_norm[n][1]} {info_list_norm[n][2]} {info_list_norm[n][3]} {id_list[n]}\n')

                frame_id += 1

                # Draw bounding boxes if visualization is enabled
                if args.visualize:
                    image = cv2.imread(imgs[i])
                    for n in range(len(info_list)):
                        color = palette.get_color(id_list[n])
                        cv2.rectangle(image, (info_list[n][0], info_list[n][1]), (info_list[n][2], info_list[n][3]), color, 2)
                        cv2.putText(image, text=str(id_list[n]), org=(info_list[n][0], info_list[n][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=3)

                    video_out.write(image)

        # Release video writer if visualization is enabled
        if args.visualize:
            video_out.release()



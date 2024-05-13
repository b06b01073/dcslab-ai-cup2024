from argparse import ArgumentParser
from collections import defaultdict
import os


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--save_dir', '-s', type=str, help='Directory containing input video frames.')
    parser.add_argument('--label_dir', '-l', type=str, help='Directory containing labels for input frames.')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    camera_labels = defaultdict(list)
    # Iterate over files in the label directory
    for file in os.listdir(args.label_dir):
        # Extract camera ID from the file name
        camera_id = int(file[0])

        # Append file path to the corresponding camera ID in the defaultdict
        camera_labels[camera_id].append(os.path.join(args.label_dir, file))

    # Sort the labels for each camera
    for k, v in camera_labels.items():
        v.sort()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # Iterate over cameras
    for cam in range(8):
        # Create subdirectory for each camera
        save = os.path.join(args.save_dir, f'{cam}')
        if not os.path.exists(save):
            os.mkdir(save)
        
        # Get the labels for the current camera
        labels = camera_labels[cam]
        
        # Iterate over each label file
        for i in range(len(labels)):
            # Open label file for reading
            label = open(labels[i], 'r')

            # Create new label file for writing in the save directory
            f = open(f'{save}/{cam}_{i:05}.txt', 'w')
            
            # Read each line from the label file and write it to the new file
            info = label.readline()
            while info:
                f.write(f'{info}')
                info = label.readline()

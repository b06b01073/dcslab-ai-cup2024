from collections import defaultdict
import os

class FrameLoader():
    def __init__(self, frame_dir, label_dir):
        
        """
        Initialize FrameLoader with directories for frames and labels.

        Args:
        - frame_dir (str): Directory containing video frames.
        - label_dir (str): Directory containing labels for frames.
        """
        
        self.frame_dir = frame_dir
        self.label_dir = label_dir

        
    def load(self, cam):
        
        """
        Load frames and labels for a specific camera.

        Args:
        - cam (int): Camera ID.

        Returns:
        - frames (list): List of paths to frame images.
        - labels (list): List of paths to label files.
        """
         
        camera_tracks = defaultdict(list)
        for file in os.listdir(self.frame_dir):
            camera_id = int(file[0])
            camera_tracks[camera_id].append(os.path.join(self.frame_dir, file))
        
        for k, v in camera_tracks.items():
            v.sort()

        camera_labels = defaultdict(list)
        for file in os.listdir(self.label_dir):
            camera_id = int(file[0])
            camera_labels[camera_id].append(os.path.join(self.label_dir, file))
        for k, v in camera_labels.items():
            v.sort()

        return camera_tracks[cam], camera_labels[cam]

    
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.io import read_image

class Cropper():

    def __init__(self, img_width):
        self.img_width = img_width

    def convert(self, W, H, x_center_norm, y_center_norm, w_norm, h_norm):

        """
        Convert normalized bounding box coordinates to absolute coordinates.

        Args:
        - W (int): Width of the original image.
        - H (int): Height of the original image.
        - x_center_norm (float): Normalized x-coordinate of the bounding box center.
        - y_center_norm (float): Normalized y-coordinate of the bounding box center.
        - w_norm (float): Normalized width of the bounding box.
        - h_norm (float): Normalized height of the bounding box.

        Returns:
        - left (int): Left coordinate of the bounding box.
        - top (int): Top coordinate of the bounding box.
        - w (int): Width of the bounding box.
        - h (int): Height of the bounding box.
        """

        x_center = x_center_norm * W
        y_center = y_center_norm * H
        w = int(w_norm * W)
        h = int(h_norm * H)
        left = int(x_center - (w/2))
        top =  int(y_center - (h/2))
        
        return left, top, w, h

    def get_image_info(self, info):

        """
        Extract bounding box information from label file.

        Args:
        - info (list): List containing bounding box information.

        Returns:
        - x_center_norm (float): Normalized x-coordinate of the bounding box center.
        - y_center_norm (float): Normalized y-coordinate of the bounding box center.
        - w_norm (float): Normalized width of the bounding box.
        - h_norm (float): Normalized height of the bounding box.
        """

        x_center_norm = float(info[1])
        y_center_norm = float(info[2])
        w_norm = float(info[3])
        h_norm = float(info[4])
        car_id = int(info[5])
        
        return x_center_norm, y_center_norm, w_norm, h_norm, car_id


    def crop_frame(self, image_path, label_path):

        """
        Crop regions of interest from the image based on bounding box information.

        Args:
        - image_path (str): Path to the image file.
        - label_path (str): Path to the label file containing bounding box information.

        Returns:
        - cropped_regions (tensor): Cropped regions of interest.
        - info_list (list): List containing bounding box coordinates for each cropped region.
        - info_list_norm (list): List containing normalized bounding box coordinates for each cropped region.
        """

        image = read_image(image_path)
        H = image.shape[1]
        W = image.shape[2]

        #get bounding box info
        label = open(label_path, 'r')
        info_list = []
        info_list_norm = []
        info = label.readline()

        cropped_regions = torch.empty((0, 3, self.img_width, self.img_width))
        
        while info:
            info = info.split(' ')
            x_center_norm, y_center_norm, w_norm, h_norm, car_id = self.get_image_info(info)
            left, top, w, h = self.convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm)
            info_list.append([left, top, left+w, top+h, car_id])
            info_list_norm.append([x_center_norm, y_center_norm, w_norm, h_norm, car_id])
            # Crop the image region based on the bounding box coordinates
            croped_img = (F.crop(image, top, left, h, w))/255
            transform = transforms.Compose([
                transforms.Resize((self.img_width, self.img_width))
            ])
            croped_img = transform(croped_img)
            cropped_regions = torch.cat((cropped_regions, torch.unsqueeze(croped_img, 0)))
            info = label.readline()
        
        return cropped_regions, info_list, info_list_norm
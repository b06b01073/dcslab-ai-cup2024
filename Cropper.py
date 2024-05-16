import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.io import read_image
import cv2
from PIL import Image
import torchvision
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

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
        
        return x_center_norm, y_center_norm, w_norm, h_norm


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
            x_center_norm, y_center_norm, w_norm, h_norm = self.get_image_info(info)
            left, top, w, h = self.convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm)
            info_list.append([left, top, left+w, top+h])
            info_list_norm.append([x_center_norm, y_center_norm, w_norm, h_norm])
            # Crop the image region based on the bounding box coordinates
            croped_img = (F.crop(image, top, left, h, w))/255
            transform = transforms.Compose([
                transforms.Resize((self.img_width, self.img_width))
            ])
            croped_img = transform(croped_img)
            cropped_regions = torch.cat((cropped_regions, torch.unsqueeze(croped_img, 0)))
            info = label.readline()
        
        return cropped_regions, info_list, info_list_norm
    
    def get_heatmap_from_crop_frame(self, cropped_regions, model, device):
        """
        Generate heatmaps from cropped regions and extract the most relevant parts based on Class Activation Maps (CAMs).

        Args:
        - cropped_regions (tensor): Tensor containing the cropped regions of interest from the original image.
        - model (torch.nn.Module): The pre-trained model used for generating heatmaps.
        - device (torch.device): The device (CPU or GPU) to perform computations on.

        Returns:
        - heatmap_cropped_regions (tensor): Tensor containing the heatmap-cropped regions, resized to the original dimensions.
        """
        # Extract the backbone of the model for generating heatmaps
        model = model.backbone

        # Initialize an empty tensor to store heatmap-cropped regions
        heatmap_cropped_regions = torch.empty((0, 3, self.img_width, self.img_width))
        
        # Define target layers for the CAM (Class Activation Map) generation
        target_layers = [model.layer4[-1]]  # Using the last layer of the fourth block
        
        # Transform to resize the cropped image to a larger size (3 times the original size)
        transform_large_size = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224*3, 224*3))
        ])

        # Transform to convert images to tensors and normalize them
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transform to resize images back to the original size
        transform_224_224 = torchvision.transforms.Compose([
            transforms.Resize((self.img_width, self.img_width))
        ])

        # Iterate through each cropped region
        for i in range(cropped_regions.size(0)):
            croped_img = cropped_regions[i].cpu()  # Move the cropped image to CPU

            # Prepare the image for CAM generation
            heatmap_img = croped_img.numpy()  # Convert the tensor to a numpy array
            heatmap_img = heatmap_img.transpose(1, 2, 0)  # Transpose to get the correct shape
            heatmap_img = Image.fromarray((heatmap_img * 255).astype(np.uint8))  # Convert to an image
            heatmap_img = transform_large_size(heatmap_img)  # Resize the image
            heatmap_img = np.array(heatmap_img)  # Convert back to a numpy array
            input_tensor = transform(heatmap_img).unsqueeze(0).to(device)  # Transform and move to device

            # Generate the CAM (Class Activation Map)
            cam = EigenCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]  # Get the first (and only) grayscale CAM
            grayscale_cam = (grayscale_cam * 255).astype(np.uint8)  # Scale CAM to 0-255

            # Threshold the CAM to get a binary mask
            _, binary_mask = cv2.threshold(grayscale_cam, 64, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv2.contourArea)  # Get the largest contour

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Resize the cropped image to a larger size
            croped_img = transform_large_size(croped_img)

            # Crop the image using the bounding box coordinates
            heatmap_crop = F.crop(cropped_regions[i], y, x, h, w)

            # Resize the cropped heatmap back to the original size
            heatmap_crop = transform_224_224(heatmap_crop)

            # Append the processed heatmap crop to the tensor
            heatmap_cropped_regions = torch.cat((heatmap_cropped_regions, torch.unsqueeze(heatmap_crop, 0)))

        return heatmap_cropped_regions
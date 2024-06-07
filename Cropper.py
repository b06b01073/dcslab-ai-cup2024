import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.io import read_image
from shapely.geometry import Polygon

class Cropper():

    def __init__(self, img_width, cam, min_size):
        """
        Initialize the Cropper class with image width, camera ID, and minimum size for bounding boxes.

        Args:
        - img_width (int): Width to which the cropped image regions will be resized.
        - cam (int): Camera ID to determine specific zones for bounding box inclusion.
        - min_size (int): Minimum area size for bounding boxes to be considered.
        """

        self.img_width = img_width
        self.cam = cam
        self.min_size = min_size


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

    def get_image_info(self, info, multi=False):

        """
        Extracts bounding box information from a label file.

        Args:
        - info (list): A list containing bounding box information.
        - multi (bool, optional): If True, returns additional ID information. Defaults to False.

        Returns:
        - tuple: If multi is False, returns a tuple containing:
            - x_center_norm (float): Normalized x-coordinate of the bounding box center.
            - y_center_norm (float): Normalized y-coordinate of the bounding box center.
            - w_norm (float): Normalized width of the bounding box.
            - h_norm (float): Normalized height of the bounding box.
        If multi is True, returns a tuple containing:
            - x_center_norm (float): Normalized x-coordinate of the bounding box center.
            - y_center_norm (float): Normalized y-coordinate of the bounding box center.
            - w_norm (float): Normalized width of the bounding box.
            - h_norm (float): Normalized height of the bounding box.
            - id_ (float): ID information of the bounding box.
        """

        x_center_norm = float(info[1])
        y_center_norm = float(info[2])
        w_norm = float(info[3])
        h_norm = float(info[4])
        
        if multi:
            id_ = float(info[5])
            return x_center_norm, y_center_norm, w_norm, h_norm, id_
        else:
            return x_center_norm, y_center_norm, w_norm, h_norm
    
    def inZone(self,left,top,w,h):
        """
        Check if the bounding box is within predefined zones for a specific camera.

        Args:
        - left (int): Left coordinate of the bounding box.
        - top (int): Top coordinate of the bounding box.
        - w (int): Width of the bounding box.
        - h (int): Height of the bounding box.

        Returns:
        - bool: True if the bounding box is within the zone, False otherwise.
        """
        cam0 = [[(0,0), (0,128), (98,186), (348,118), (430,115),(668,0)], [(747,0), (690,145), (730,148), (760,0)], [(816,0), (1083,195), (1280,234),(1280,0)]]
        cam1 = [[(0,0), (0,137), (1024,276), (1280,248)], [(0,172), (0,219), (719,396), (824,358)]]
        cam2 = [[(0,0), (0,322), (81,298), (422,98), (422,0)], [(476,0), (484,308), (611,295), (543,163), (543,0)], [(562,0), (562,99), (1077,299), (1280,325), (1280,0)]]
        cam3 = [[(0,0), (0,345), (260,267), (260,0)], [(5,720), (340,272), (439.720)], [(366,0), (366,272), (1280,476), (1280,0)]]
        cam4 = [[(0,0), (0,356), (728,0)], [(783,0), (620,363), (734,366)], [(857,0), (1280,356), (1280,0)]]
        cam5 = [[(0,0), (0,283), (74,265), (58,0)], [(0,330), (0,642), (209,590), (109,324)], [(58,0), (68,183), (666,276), (1280,237), (1280,0)], [(731,308), (1280,428), (1280,278)]]
        cam6 = [[(0,0), (0,170), (1280,163), (1280,0)], [(0,200), (0,278), (164,195)], [(712,183), (1058,326), (1280,337), (1280,190)]]
        cam7 = [[(0,97), (0,327), (260,175)], [(310,190), (312,720), (452,720)], [(0,0), (1280,478), (0,97), (1280,478), (1280,0)]]

        zones = [cam0, cam1, cam2, cam3, cam4, cam5, cam6, cam7]

        bbox = Polygon([(left, top), (left+w, top), (left+w, top+h), (left, top+h)])
        for polygon in zones[self.cam]:
            poly = Polygon(polygon)
            if poly.contains(bbox):
                return True
            else:
                return False
    
    def sizeCheck(self, w, h):
        """
        Check if the bounding box meets the minimum size requirement.

        Args:
        - w (int): Width of the bounding box.
        - h (int): Height of the bounding box.

        Returns:
        - bool: True if the bounding box meets the minimum size requirement, False otherwise.
        """
        if w * h >= self.min_size:
            return True
        else:
            return False


    def crop_frame(self, image_path, label_path, multi=False):

        """
        Crop regions of interest from the image based on bounding box information.

        This function reads an image and its corresponding label file, extracts bounding box 
        information, and crops regions of interest from the image. It checks if the bounding 
        boxes are within certain zones and if they meet size criteria.
        Args:
        - image_path (str): Path to the image file.
        - label_path (str): Path to the label file containing bounding box information.
        - multi (bool, optional): If True, returns additional ID information for each bounding 
        box. Defaults to False.

        Returns:
        - cropped_regions (torch.Tensor): Tensor containing cropped regions of interest.
        - info_list (list): List containing bounding box coordinates for each cropped region.
        - info_list_norm (list): List containing normalized bounding box coordinates for each 
        cropped region.
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
            if multi:
                x_center_norm, y_center_norm, w_norm, h_norm, id_ = self.get_image_info(info, multi)
            else:
                x_center_norm, y_center_norm, w_norm, h_norm = self.get_image_info(info)
            left, top, w, h = self.convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm)
            
            if not self.inZone(left, top, w, h) and self.sizeCheck(w, h):
                if multi:
                    info_list.append([left, top, left+w, top+h, id_])
                    info_list_norm.append([x_center_norm, y_center_norm, w_norm, h_norm, id_])
                else:
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
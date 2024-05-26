from directionModel import directionClassification
from Cropper import Cropper
import torch.nn as nn
import torchvision
import os
import torch
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        # in[N, 3, 32, 32] => out[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        # in[N, 16, 16, 16] => out[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2)

        )
        # in[N, 32 * 8 * 8] => out[N, 128]
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 8 * 4, 256),
            nn.ReLU(True)
        )
        # in[N, 128] => out[N, 64]
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        # in[N, 64] => out[N, 10]
        self.out = nn.Linear(65, self.num_classes)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     output = self.out(x)
    #     return output
    # def forward(self, x, w, h):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     w = torch.unsqueeze(torch.unsqueeze(torch.tensor(w).to(device),0),1)
    #     h = torch.unsqueeze(torch.unsqueeze(torch.tensor(h).to(device),0),1)
        
        
    #     x = torch.cat((x,w),dim=1)
    #     x = torch.cat((x,h),dim=1)
    #     #print(x.shape)
    #     #print(x.shape)
    #     output = self.out(x)
    #     return output
    def forward(self, x, w, h):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        handw = h/w
        #print(handw)
        handw = torch.unsqueeze(torch.unsqueeze(torch.tensor(handw).to(device),0),1)
        #print(x.shape, w.shape, h.shape)
        x = torch.cat((x,handw),dim=1)
        #print(x.shape)
        output = self.out(x)
        return output
    
def multiCam(camera_num):
    '''
    This function can choose the cars that will cross camera.

    Args:
    - camera_num: The camera number you want to filter.

    Returen:
    -carToNext: The dictionary that can find the car that may exist in the next camera.
    '''
    carToNext = {}
    count = 0
    for i in range(0, 8):
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
    if camera_num == -1:
        for j in range(0, 8):
            count = 0
            Forward(j, carToNext)
            Backward(j ,carToNext)
        print(carToNext)
    else:
        Forward(camera_num, carToNext)
        Backward(camera_num ,carToNext)
    return carToNext
def ImgandLabel(j, k):
    if k < 10:
        tag = "_0000"
    elif k < 100:
        tag = "_000"
    else:
        tag = "_00"
    image_path = "./test_set/IMAGE/1016_150000_151900/" + str(j) + tag + str(k) + ".jpg"
    label_path = "./test_set/LABEL/1016_150000_151900/" + str(j) + tag + str(k) + ".txt"
    return image_path, label_path

def GetBBofEachFrame(Forward, camera, img, info_list, info_list_norm, j, k, carToNext, previousDirection, previousReject, previousX, previousY):
        
    count = 0
    for i in range(0, img.size(dim=0)):
        car_id = info_list[i][4]
        x = info_list_norm[i][0]
        y = info_list_norm[i][1]
        w = info_list[i][2]
        h = info_list[i][3]
        if j > 0 and car_id in carToNext:
            if carToNext[car_id] == j-2:
                count+=1
                print(car_id, carToNext[car_id], j, k)
            
        direction = directionClassification(img[i], w, h)
        # if(car_id == 4474 or car_id == 4486 or car_id == 4476):
        #     torchvision.utils.save_image(img[i], str(direction[0].item()) + "/" + str(car_id)+ "-"+str(j)+"-"+str(k)+'.jpg') 
        check=True
        
        if car_id in previousDirection:
            check = direction_check(previousDirection[car_id], direction, previousReject[car_id], previousX[car_id], previousY[car_id], int(100*x), int(100*y))
        # if car_id == 4441:
        #     print(check)
        if check == True:
            next_camera = toCamera(Forward, camera, direction[0].item())
            carToNext[car_id] = next_camera
            previousReject[car_id] = -1
        else:
            previousReject[car_id] = direction[0].item()
            
        previousX[car_id] = int(x*100)
        previousY[car_id] = int(y*100)   
        
        #print(j, k, car_id, direction[0].item(), next_camera)
        previousDirection[car_id] = direction[0].item()
        if car_id not in previousReject:
            previousReject[car_id] = -1
            

        # if(car_id ==4474 or car_id == 4476 or car_id == 4486):
        #     print(car_id, direction, carToNext[car_id], k)
           # print(previousReject[car_id], previousDirection[car_id])
       
def Backward(j, carToNext):
    print("------camera_"+str(j)+"_Backwarding------")
    previousDirection = {}
    previousReject = {}
    previousX = {}
    previousY = {}
    tmpDict = {}
    for k in range(360, 0, -1):
        image_path, label_path = ImgandLabel(j , k)
        Crop = Cropper(224)
        img, info_list, info_list_norm = Crop.crop_frame(image_path, label_path, True)
        camera = j
        GetBBofEachFrame(False, camera, img, info_list, info_list_norm, j, k, tmpDict, previousDirection, previousReject, previousX, previousY)
    for key in tmpDict.keys():
        if carToNext[key] == j+1 or tmpDict[key] == j+1:
            carToNext[key] = j+1

def Forward(j, carToNext):
    print("------camera_"+str(j)+"_Forwarding------")
    previousDirection = {}
    previousReject = {}
    previousX = {}
    previousY = {}
    for k in range(1, 361):
        image_path, label_path = ImgandLabel(j , k)
        Crop = Cropper(224)
        img, info_list, info_list_norm = Crop.crop_frame(image_path, label_path, True)
        camera = j
        GetBBofEachFrame(True, camera, img, info_list, info_list_norm, j, k, carToNext, previousDirection, previousReject, previousX, previousY)
        
def direction_check(previous, current, rej, preX, preY, cuX, cuY):
    down = 0 
    up = 1
    left = 2
    right = 3
    right_up = 4
    right_down = 5
    left_up = 6
    left_down = 7
    if preX == cuX and preY == cuY:
        return False
    if rej == current:
        return True
    # if previous == down:
    #     if current == left_down or current == right_down:
    #         return True
    # elif previous == left_down:
    #     if current == down and current == left:
    #         return True
    # elif previous == left:
    #     if current == left_down and current == left_up:
    #         return True
    # elif previous == left_up:
    #     if current == left or current == up:
    #         return True
    # elif previous == up:
    #     if current == left_up and current == right_up:
    #         return True
    # elif previous == right_up:
    #     if current == up or current  == right:
    #         return True
    # elif previous == right:
    #     if current == right_down or current  == right_up:
    #         return True
    # elif previous == right_down:
    #     if current == right or current  == down:
    #         return True
    # return False
    if previous == down:
        if current == up or current == left_up or current == right_up:
            return False
    elif previous == left_down:
        if current == up or current  == right_up or current == right:
            return False
    elif previous == left:
        if current == right_up or current  == right or current == right_down:
            return False
    elif previous == left_up:
        if current == right or current  == right_down or current == down:
            return False
    elif previous == up:
        if current == right_down or current  == down or current == left_down:
            return False
    elif previous == right_up:
        if current == down or current  == left_down or current == left:
            return False
    elif previous == right:
        if current == left_down or current  == left_up or current == left:
            return False
    elif previous == right_down:
        if current == up or current  == left_up or current == left:
            return False
    return True
       
def toCamera(Forward, current_camera, direction):
    down = 0 
    up = 1
    left = 2
    right = 3
    right_up = 4
    right_down = 5
    left_up = 6
    left_down = 7
    if Forward == True:
        if current_camera == 0:
            if direction == up or direction == left_up:
                return 1
        elif current_camera == 1:
            if direction == right or direction == right_down:
                return 2
        elif current_camera == 2:
            if direction == down or direction == left_down or direction == right_up:
                return 3
        elif current_camera == 3:
            if direction == down or direction == left_down:
                return 4
        elif current_camera == 4:
            if direction == up:
                return 5 
        elif current_camera == 5:
            if direction == left or direction == left_down or direction == right_down:
                return 6
        elif current_camera == 6:
            if direction == left:
                return 7
        else:
            return 8
    else:
        if current_camera == 0:
            if direction == down or direction == left_down:
                return 1
        elif current_camera == 1:
            if direction == left or direction == left_up:
                return 2
        elif current_camera == 2:
            if direction == up or direction == left_up or direction == right_up:
                return 3
        elif current_camera == 3:
            if direction == up or direction == left_up:
                return 4
        elif current_camera == 4:
            if direction == down or direction == left_down:
                return 5 
        elif current_camera == 5:
            if direction == right_up or direction == right:
                return 6
        elif current_camera == 6:
            if direction == right:
                return 7
        else:
            return 8
    return current_camera-1        
        

# multiCam(-1)
# img = Image.open("./4/4431-2-148.jpg")
# w, h = img.size
# direction = directionClassification(img, w, h)
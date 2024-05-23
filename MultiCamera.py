from directionModel import directionClassification
from Cropper import Cropper
import torch.nn as nn
import torchvision
import os
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
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.out(x)
        return output
    
def multiCam(camera_num):
    carToNext = {}
    for i in range(0, 8):
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
    if camera_num == -1:
        for j in range(0, 8):
            for k in range(1, 361):
                if k < 10:
                    tag = "_0000"
                elif k < 100:
                    tag = "_000"
                else:
                    tag = "_00"
                image_path = "./images/" + str(j) + tag + str(k) + ".jpg"
                label_path = "./labels/" + str(j) + tag + str(k) + ".txt"
                Crop = Cropper(224)
                img, info_list, info_list_norm = Crop.crop_frame2(image_path, label_path)
                #print(info_list)
                #break
                camera = j
                #print(img.size(dim=0))
                for i in range(0, img.size(dim=0)):
                    '''        
                    'down'      :(0)
                    'up'        :(1)
                    'left'      :(2)
                    'right'     :(3)
                    'right_up'  :(4)
                    'right_down':(5)
                    'left_up'   :(6)
                    'left_down' :(7)
                    '''
                    car_id = info_list[i][4]
                    if j > 0 and car_id in carToNext:
                        if carToNext[car_id] == j or carToNext[car_id] > j:
                            print(car_id)
                        else:
                            print(car_id, carToNext[car_id])
                    #print(j, k, car_id)
                    direction = directionClassification(img[i])
                    #print(direction[0])
                    
                    torchvision.utils.save_image(img[i], str(direction[0].item()) + "/" + str(car_id)+'.jpg') 
                    #print(camera, direction[0].item())
                    next_camera = toCamera(camera, direction[0].item())
                    
                    #print(j, k, car_id, direction[0].item(), next_camera)
                    carToNext[car_id] = next_camera
                    # if(car_id == 20):
                    #     print(direction, next_camera)
                    # try:
                    #     carToNext[car_id] = next_camera
                    # except:
                    #     while True:
                    #         try: 
                    #             carToNext[car_id] = next_camera
                    #             break
                    #         except:
                    #             carToNext.append(0)
            
        print(carToNext)
            # for z in range(0,8):
            #     print(carToNext.count(z))
    else:
        for k in range(1, 361):
            if k < 10:
                tag = "_0000"
            elif k < 100:
                tag = "_000"
            else:
                tag = "_00"
            image_path = "./images/" + str(camera_num) + tag + str(k) + ".jpg"
            label_path = "./labels/" + str(camera_num) + tag + str(k) + ".txt"
            Crop = Cropper(224)
            img, info_list, info_list_norm = Crop.crop_frame2(image_path, label_path)
            #print(info_list)
            #break
            camera = camera_num
            #print(img.size(dim=0))
            for i in range(0, img.size(dim=0)):
                '''        
                'down'      :(0)
                'up'        :(1)
                'left'      :(2)
                'right'     :(3)
                'right_up'  :(4)
                'right_down':(5)
                'left_up'   :(6)
                'left_down' :(7)
                '''
                car_id = info_list[i][4]
                #print(j, k, car_id)
                direction = directionClassification(img[i])
                #print(direction[0])
                
                torchvision.utils.save_image(img[i], str(direction[0].item()) + "/" + str(car_id)+'.jpg') 
                #print(camera, direction[0].item())
                next_camera = toCamera(camera, direction[0].item())
                
                #print(j, k, car_id, direction[0].item(), next_camera)
                carToNext[car_id] = next_camera
                # try:
                #     carToNext[car_id] = next_camera
                # except:
                #     while True:
                #         try: 
                #             carToNext[car_id] = next_camera
                #             break
                #         except:
                #             carToNext.append(0)

        print(carToNext)
        # for z in range(0,8):
        #     print(carToNext.count(z))
    return carToNext
    
def toCamera(current_camera, direction):
    down = 0 
    up = 1
    left = 2
    right = 3
    right_up = 4
    right_down = 5
    left_up = 6
    left_down = 7
    if current_camera == 0:
        if direction == up or direction == left_up:
            return 1
    elif current_camera == 1:
        if direction == right or direction == right_down:
            return 2
    elif current_camera == 2:
        if direction == down or direction == left_down:
            return 3
    elif current_camera == 3:
        if direction == down or direction == left_down:
            return 4
    elif current_camera == 4:
        if direction == up:
            return 5 
    elif current_camera == 5:
        if direction == right or direction == right_down:
            return 6
    elif current_camera == 6:
        if direction == left:
            return 7
    else:
        return 8
    return -1
                
        

multiCam(0)

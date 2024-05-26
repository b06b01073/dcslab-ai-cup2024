import torch as torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random


DOANLOAD_DATASET = True
LR = 0.001
BATCH_SIZE=128
EPOCH = 50
MODELS_PATH = './models'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, 4),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([64,64]),
    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([64,64]),
    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
classes = ["down", "up", "left", "right", "right_up", "right_down", "left_up", "left_down"]
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
    #     w = torch.unsqueeze(torch.tensor(w.to(device)),0)
    #     h = torch.unsqueeze(torch.tensor(h.to(device)),0)
    #     #print(x.shape, w.shape, h.shape)
    #     x = torch.cat((x,w),dim=1)
    #     x = torch.cat((x,h),dim=1)
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
        handw = torch.unsqueeze(torch.tensor(handw),1)
        #print(x.shape, w.shape, h.shape)
        x = torch.cat((x,handw),dim=1)
        #print(x.shape)
        output = self.out(x)
        return output
def model_train():
    class dataSets(Dataset):
        def __init__(self, transform=None):
            self.root_dir = ["./datasets2/down/", "./datasets2/up/","./datasets2/Left/","./datasets2/Right/", 
            "./datasets2/Right up/", "./datasets2/Right down/","./datasets2/Left up/","./datasets2/Left down/"]
            self.labels = ["down", "up", "left", "right", "right_up", "right_down", "left_up", "left_down"]
            self.N_Pic = [2737, 2686, 3746 ,2947]
            #self.NPic = [207, 313, 462, 261]
            
            self.transform = transform
            self.P=[]
            self.L=[]
            self.f = open('out2.txt','r')
            self.dataRoots=self.f.read().split('\n')[:-1]
            #print(type(self.dataRoots))
            
            for i in range(0,8):
                # temp_record=[]
                # for j in range(0,self.N_Pic[i]):
                #     te = random.randint(0, self.N_Pic[i])
                #     #print(os.path.join(self.root_dir[i]+str(te)+'.jpg'))
                #     while(te not in temp_record and os.path.join(self.root_dir[i]+str(te)+'.jpg') not in self.dataRoots):
                #         te = random.randint(0, self.N_Pic[i])
                #     temp_record.append(te)
                #     self.P.append(self.root_dir[i]+str(te)+'.jpg')
                #     self.L.append(self.labels[i])
                temp_record=[]
                for j in range(0, 450):
                    te = random.randint(0, 450)
                    #print(os.path.join(self.root_dir[i]+str(te)+'.jpg'))
                    while(te in temp_record and os.path.join(self.root_dir[i]+str(te)+'.jpg') not in self.dataRoots):
                        te = random.randint(0, 450)
                    temp_record.append(te)
                    self.P.append(self.root_dir[i]+str(te)+'.jpg')
                    self.L.append(self.labels[i])
            self.n_samples=len(self.P)
            
        def __getitem__(self, index):
            img_path = self.P[index%self.n_samples]
            label = self.L[index%self.n_samples]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        def __len__(self):
            return self.n_samples
    class testDataSets(Dataset):
        def __init__(self, transform=None):
            self.root_dir = './datasets/test/'
            self.labels = ["down", "up", "left", "Right"]
            self.NPic = [23, 40, 52, 65]
            
            self.transform = transform
            self.P=[]
            self.L=[]
            j = 0
            for i in range(0,self.NPic[3]):
                self.P.append(self.root_dir+str(i)+'.jpg')
                self.L.append(self.labels[j])
                if i == self.NPic[j]:
                    j += 1
            self.n_samples=len(self.P)
            
        def __getitem__(self, index):
            img_path = self.P[index%self.n_samples]
            label = self.L[index%self.n_samples]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        def __len__(self):
            return self.n_samples   
    train_size = 0.8
    test_size = 0.2
    #train_dataset, test_dataset = torch.utils.data.random_split(dataSets, [train_size, test_size])
    data = dataSets(
        transform=train_transform,
    )

    train_data, test_data = torch.utils.data.random_split(data, [0.9, 0.1])

    # train_data = dataSets(
    #     transform=train_transform,
    # )

    # test_data = testDataSets(
    #     transform=test_transform,
    # )

    data_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    #classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    cnn = CNN(len(classes))
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        cnn.train()
        for step, (x, y) in enumerate(data_loader):
            tem_label=[]
            for i in y:
                if i == 'down':
                    tem_label.append(0)
                elif i == 'up':
                    tem_label.append(1)
                elif i == 'left':
                    tem_label.append(2)
                elif i == 'right':
                    tem_label.append(3)
                elif i == 'right_up':
                    tem_label.append(4)
                elif i == 'right_down':
                    tem_label.append(5)
                elif i == 'left_up':
                    tem_label.append(6)
                else:
                    tem_label.append(7)
            #print(tem_label)
            b_x = Variable(x, requires_grad=False).to(device)
            b_y = Variable(torch.tensor(tem_label), requires_grad=False).to(device)
            out = cnn(b_x)
            loss = loss_function(out, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: {} | Step: {} | Loss: {}'.format(epoch + 1, step, loss))
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=100,
            shuffle=False
        )
        test_x, test_y = next(iter(test_loader))
        tem_label = []
        for i in test_y:
            if i == 'down':
                tem_label.append(0)
            elif i == 'up':
                tem_label.append(1)
            elif i == 'left':
                tem_label.append(2)
            elif i == 'right':
                tem_label.append(3)
            elif i == 'right_up':
                tem_label.append(4)
            elif i == 'right_down':
                tem_label.append(5)
            elif i == 'left_up':
                tem_label.append(6)
            else:
                tem_label.append(7)
        test_y = torch.tensor(tem_label).to(device)
        text_x = test_x.to(device)
        cnn.eval()
        prediction = torch.argmax(cnn(test_x.to(device)), 1)
        acc = torch.eq(prediction, test_y)
        k = 0

        print('Accuracy: {:.2%}'.format((torch.sum(acc) / acc.shape[0]).item()))
        print(prediction)
        print(test_y)

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
        torch.save(cnn, os.path.join(MODELS_PATH, 'cnn_model.pt'))

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=100,
        shuffle=False
    )
    test_x, test_y = next(iter(test_loader))
    tem_label = []
    for i in test_y:
        if i == 'down':
            tem_label.append(0)
        elif i == 'up':
            tem_label.append(1)
        elif i == 'left':
            tem_label.append(2)
        elif i == 'right':
            tem_label.append(3)
        elif i == 'right_up':
            tem_label.append(4)
        elif i == 'right_down':
            tem_label.append(5)
        elif i == 'left_up':
            tem_label.append(6)
        else:
            tem_label.append(7)
    test_y = torch.tensor(tem_label).to(device)
    text_x = test_x.to(device)
    cnn.eval()
    prediction = torch.argmax(cnn(test_x.to(device)), 1)
    for i in range(0, 8):
        os.mkdir(str(i))
    j=0
    for x, y in zip(text_x, prediction):
        torchvision.utils.save_image(x, str(y.item()) + '/' + str(j)+'.jpg')   
        j = j+1
    acc = torch.eq(prediction, test_y)
    k = 0

    print('Accuracy: {:.2%}'.format((torch.sum(acc) / acc.shape[0]).item()))
    print(prediction)
    print(test_y)

def directionClassification(img, w, h): 
    #print(type(img))
    
    img = torchvision.transforms.ToPILImage()(img)
    
    #print(type(img))
    cnn = CNN(len(classes))
    cnn = torch.load("./models/cnn_model_93_1_65.pt")
    cnn = cnn.to(device)
    
    cnn.eval()
    #print(img.shape)
    #print(nn.Softmax()(cnn(test_transform(img).unsqueeze(dim=0).to(device), w, h)))
    prediction = torch.argmax(cnn(test_transform(img).unsqueeze(dim=0).to(device), w, h), 1)
    return prediction
# Readme

## Rules of conduct

Read this [page](https://hackmd.io/@2aRuhhznQfOr_IvFkBUYKQ/SJ0KESMzR) first before you start working on this project.

## 簡介
出於意外把所有東西都推上來了
主要有三個檔案 Multicamera.py, directiionmodel.py, cropper.py
本repo主要使用於車子方向的判別與過濾每一台camera需要比對的embedding

## 本repo 用法
1. git clone
```
git clone https://github.com/Jonas0126/dcslab-ai-cup2024.git
```
2. 準備資料集，目錄結構如下:
```
LABEL
└── 0902_150000_151900
    ├── 0_00001.TXT
    ├── 0_00002.TXT
    ├── 0_00003.TXT
    ├── 0_00004.TXT
    ├── 0_00005.TXT
    ├── 0_00006.TXT
    ├── 0_00007.TXT
    ├── 0_00008.TXT
    ├── 0_00009.TXT
    └── 0_00010.TXT
    └── ...
```
```
IMAGE
└── 0902_150000_151900
    ├── 0_00001.jpg
    ├── 0_00002.jpg
    ├── 0_00003.jpg
    ├── 0_00004.jpg
    ├── 0_00005.jpg
    ├── 0_00006.jpg
    ├── 0_00007.jpg
    ├── 0_00008.jpg
    ├── 0_00009.jpg
    └── 0_00010.jpg
    └── ...
```
3. 呼叫 MultiCamera.py 中 multicam.py
```
mutlicam(camera_num)
```
  * --camera_num 假設現在進行camera N的比對，則camera_num請傳N-1
4. 回傳值 N-1 camera中所有車輛的id及其到哪個camera
```
carToNext = {} 此為一個dictionary
```
使用方式
```
current_camera = carToNext[car_id]
```



# Readme

## Rules of conduct

Read this [page](https://hackmd.io/@2aRuhhznQfOr_IvFkBUYKQ/SJ0KESMzR) first before you start working on this project.

# Readme

## Rules of conduct

Read this [page](https://hackmd.io/@2aRuhhznQfOr_IvFkBUYKQ/SJ0KESMzR) first before you start working on this project.

## 簡介
該repo實現了單鏡頭多物體追蹤，接收監視器的每一幀作為輸入，並輸出每幀畫面中每個物體的邊界框座標和 ID。輸出格式遵循 AI-CUP 數據集的結構。

## 用法
1. git clone
2. 準備輸入目錄，目錄結構如下:
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
3. 執行 main.py
```
python main.py -f IMAGE/0902_150000_151900/ -l LABEL/0902_150000_151900/ --out result_label --model resnet101_ibn_a --buffer_size 1 --threshold 0.5 --lambda_value 0.3
```
  * --frame_dir：輸入畫面的目錄。
  * --label_dir：輸入標籤的目錄。
  * --model：用於特徵提取的模型的名稱（默認為 'resnet101_ibn_a'）。
  * --out：保存輸出的目錄。
  * --width：裁剪圖片的寬度（默認為 256）。
  *  --buffer_size：儲存過去的frame的buffer的大小。
  * --threshold：threshold for tracking objects。
  * --lambda_value：用於re-ranking的 Lambda 值。
  * --re_rank : 是否要使用re-rank。

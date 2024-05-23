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
3. 執行 main.py
```
python main.py -f IMAGE/0902_150000_151900/ -l LABEL/0902_150000_151900/ --out aicup_ts/labels/0902_150000_151900 --cam 0
```
  * --frame_dir：輸入畫面的目錄。
  * --label_dir：輸入標籤的目錄。
  * --model：用於特徵提取的模型的名稱（默認為 'resnet101_ibn_a'）。
  * --out：保存輸出的目錄。
  * --width：裁剪圖片的寬度（默認為 224）。
  * --buffer_size：儲存過去的frame的buffer的大小。
  * --threshold：threshold for tracking objects。
  * --lambda_value：用於re-ranking的 Lambda 值。
  * --re_rank : 是否要使用re-rank。
  * --visualize : 是否要輸出影片
  * --cam : 指定要追蹤的camera編號
  * --finetune : 指定是否為fine-tune模式

輸出會儲存在指定的目錄裡，目錄結構如下所示:
```
aicup_test/labels/0902_150000_151900
├── 0
│   ├── 0_00001.txt
│   ├── 0_00002.txt
│   ├── 0_00003.txt
|   └── 0_00004.TXT
|   └── ...
├── 1
├── 2
├── 3
├── 4
├── 5
├── 6
└── 7
```
## evaluation.py 使用方法
1. 使用parseAicup.py將aicup的label根據cam編號分成8個資料夾
```
python parseAicup.py -s aicup_gt/labels/0902_150000_151900 -l LABEL/0902_150000_151900/
```
2. 將ground truth的label和預測的label轉換成MOT15格式
```
python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_gt/labels/0902_150000_151900 --MOT15_dir MOT15/aicup_gt/0902_150000_151900
```
```
python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_ts/labels/0902_150000_151900 --MOT15_dir MOT15/aicup_ts/0902_150000_151900
```
    * --AICUP_dir : 要被轉換的label的目錄
    * --MOT15_dir : 保存轉換後label的目錄
3. 評估結果
```
python tools/evaluate.py --gt_dir MOT15/aicup_gt/0902_150000_151900 --ts_dir MOT15/aicup_ts/0902_150000_151900 --mode single_cam --cam 0
```
    * --gt_dir : MOT15格式的ground truth的目錄
    * --ts_dir : MOT15格式的預測的label的目錄
    * --mode : 設定使用multi camera模式還是single camera來評價結果
    * --cam : 指定評價哪一個camera(只有在single cam模式下有用)
評估結果會儲存在ts_result目錄下:
```
ts_result
└── 0902_150000_151900
    └── 0.txt
```

## fine-tune參數使用方法
1. 執行run_docker.sh
```
./run_docker.sh
```
2. 切換環境至botsort
```
conda activate botsort
```
3. 執行track_evaluation.sh
```
./track_evaluation.sh -p threshold -s 50 -e 60 -t 1 -v False -c 0
```
* -p : 指定fine-tune的參數，[buffer_size, threshold]
* -s 、 -e 、-t : 設置參數的區間，-s代表參數開始的值，-e代表參數結束的值，-t代表步長，因為shell script無法運算浮點數，
    因此在輸入參數時需要將值都乘以100，例如threshold的區間為0.5至0.6，步長為0.01，則須輸入-s 50 -e 60 -t 1。
    
評分結果會儲存在ts_result目錄下以cam編號為名字的目錄裡面的文字檔，檔名為fine-tune的參數，如下所示:
```
ts_result
└── 7
    └── buffer_size.txt
```
文字檔裡的內容為各參數值在不同天同一個cam的平均分數:
```
buffer_size 1.0, AVE IDF1 : 0.8576504865060863, AVE MOTA : 0.9541107325987048
buffer_size 2.0, AVE IDF1 : 0.9355950572352348, AVE MOTA : 0.9698999148566754
```

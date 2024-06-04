# Readme

[![Static Badge](https://img.shields.io/badge/Ranked_%231-2024_AI_CUP_Spring_on_MTMC-blue?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNS4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjQgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZmlsbD0iI0ZGRDQzQiIgZD0iTTQuMSAzOC4yQzEuNCAzNC4yIDAgMjkuNCAwIDI0LjZDMCAxMSAxMSAwIDI0LjYgMEgxMzMuOWMxMS4yIDAgMjEuNyA1LjkgMjcuNCAxNS41bDY4LjUgMTE0LjFjLTQ4LjIgNi4xLTkxLjMgMjguNi0xMjMuNCA2MS45TDQuMSAzOC4yem01MDMuNyAwTDQwNS42IDE5MS41Yy0zMi4xLTMzLjMtNzUuMi01NS44LTEyMy40LTYxLjlMMzUwLjcgMTUuNUMzNTYuNSA1LjkgMzY2LjkgMCAzNzguMSAwSDQ4Ny40QzUwMSAwIDUxMiAxMSA1MTIgMjQuNmMwIDQuOC0xLjQgOS42LTQuMSAxMy42ek04MCAzMzZhMTc2IDE3NiAwIDEgMSAzNTIgMEExNzYgMTc2IDAgMSAxIDgwIDMzNnptMTg0LjQtOTQuOWMtMy40LTctMTMuMy03LTE2LjggMGwtMjIuNCA0NS40Yy0xLjQgMi44LTQgNC43LTcgNS4xTDE2OCAyOTguOWMtNy43IDEuMS0xMC43IDEwLjUtNS4yIDE2bDM2LjMgMzUuNGMyLjIgMi4yIDMuMiA1LjIgMi43IDguM2wtOC42IDQ5LjljLTEuMyA3LjYgNi43IDEzLjUgMTMuNiA5LjlsNDQuOC0yMy42YzIuNy0xLjQgNi0xLjQgOC43IDBsNDQuOCAyMy42YzYuOSAzLjYgMTQuOS0yLjIgMTMuNi05LjlsLTguNi00OS45Yy0uNS0zIC41LTYuMSAyLjctOC4zbDM2LjMtMzUuNGM1LjYtNS40IDIuNS0xNC44LTUuMi0xNmwtNTAuMS03LjNjLTMtLjQtNS43LTIuNC03LTUuMWwtMjIuNC00NS40eiIvPjwvc3ZnPg==)
](https://tbrain.trendmicro.com.tw/Competitions/Details/33)

## Rules of conduct

Read this [page](https://hackmd.io/@2aRuhhznQfOr_IvFkBUYKQ/SJ0KESMzR) first before you start working on this project.

## 簡介
該repo實現了單鏡頭多物體追蹤，接收監視器的每一幀作為輸入，並輸出每幀畫面中每個物體的邊界框座標和 ID。輸出格式遵循 AI-CUP 數據集的結構。

可以透過track_evaluation.sh進行fine-tune.

## 環境建立
如果是使用linux環境，輸入以下指令即可:
```
conda env create -f environment.yml
```
若為windows則依序安裝下列package:
```
conda create --name aicup2024 python=3.12
conda activate aicup2024
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas
conda install anaconda::scipy
conda install conda-forge::tqdm
conda install anaconda::xmltodict
pip install opencv-python
pip install loguru
```
若出現以下錯誤:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
執行以下指令:
```
pip uninstall opencv-python
pip install opencv-python-headless
```

環境建立完成後:
```
conda activate aicup2024
```
## single camera tracking 用法
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
若要評估單一模型執行track_evaluation.sh，track_evaluation.sh會遍歷所有模型所有日期以及相機
```
./track_evaluation.sh -p threshold -s 50 -e 60 -t 1 -v False -d m
```

若要評估集成模型執行track_evaluation._ensemble.sh，
```
./track_evaluation.sh -p threshold -s 50 -e 60 -t 1 -v False -d m
```
* -p : 指定fine-tune的參數，[buffer_size, threshold]
* -s 、 -e 、-t : 設置參數的區間，-s代表參數開始的值，-e代表參數結束的值，-t代表步長，因為shell script無法運算浮點數，
    因此在輸入參數時需要將值都乘以100，例如threshold的區間為0.5至0.6，步長為0.01，則須輸入-s 50 -e 60 -t 1。
* -v : 設定使否要輸出影片
* -d : 設定使用早上還是晚上的資料集，m代表使用早上的資料集，n代表使用晚上的資料集

若為單一模型則評分結果會儲存在ts_result目錄下以cam編號為名字的目錄裡面的文字檔，檔名為fine-tune的參數，如下所示:
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

## Multi-Camera Tracking Result
|     | ave_v1 | ave_v2 | min  | max |
| --- | ------ | ------ | ---- | --- |
|    IDF1 |    80.7    |   63.5     |   96.2   |  57.1   |
|     MOTA|    98.1| 96.4   | 99.7 |   96.4  |

* $min$ $d(C_i, C_j)$ : $\mathop{\min}_{a \in C_i, b \in C_j} d(a,b)$


* $max$ $d(C_i, C_j)$ : $\mathop{\max}_{a \in C_i, b \in C_j} d(a,b)$


* $ave\_{v1}$ $d(C_i,C_j)$ : $\sum_{a \in C_i,b \in C_j} \frac{d(a,b)}{|C_i||C_j|}$


* $ave\_v2$ $d(C_i,C_j)$ : $d(\frac{\sum_{a \in C_i}a}{|C_i|}, \frac{\sum_{b \in C_j}b}{|C_j|})$

* $d(a,b)$是Cosine Similarity
## fine-tuned models
```
# options: [resnet101_ibn_a, se_resnet101_ibn_a, densenet169_ibn_a, swin_reid, resnext101_ibn_a]
net = torch.hub.load('b06b01073/dcslab-ai-cup2024', 'resnet101_ibn_a') # you can also set use_test=True if you want to use the model trained with training + testing set

eu_embedding, cos_embedding, _ = net(processed_img) # processed_img.shape: (b, c, h, w)

# The rest is the same as the code example from veri776-pretrain repo
```

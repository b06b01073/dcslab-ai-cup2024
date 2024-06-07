date_list=(0902_130006_131041 0902_180000_181551 0903_125957_131610 0903_175958_181607 \
           0924_125953_132551 0924_175955_181603 0925_130000_131605 0925_175958_181604 \
           1001_130000_131559 1001_180000_181558 1002_130000_131600 1002_180000_181600 \
           1015_130000_131600 1015_180001_183846 1016_130000_131600 1016_180000_181600)



for date in "${date_list[@]}"
do
    for cam in {0..7}
    do
        # Start tracking the cars on the monitor.
        echo python main.py -f ../32_33_AI_CUP_testdataset/AI_CUP_testdata/images/$date -l ../detect_result/$date --out RE_RESULT/labels/$date --cam $cam --model swin_reid
        python main.py -f ../32_33_AI_CUP_testdataset/AI_CUP_testdata/images/$date -l ../detect_result/$date --out RE_RESULT/labels/$date --cam $cam --model swin_reid
    done
done
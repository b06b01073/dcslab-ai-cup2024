date_list=(0902_130006_131041 0902_180000_181551 0903_125957_131610 0903_175958_181607 \
           0924_125953_132551 0924_175955_181603 0925_130000_131605 0925_175958_181604 \
           1001_130000_131559 1001_180000_181558 1002_130000_131600 1002_180000_181600 \
           1015_130000_131600 1015_180001_183846 1016_130000_131600 1016_180000_181600)

for date in "${date_list[@]}";
    do
        echo python multi_match.py --date $date --model swin_reid --mode min --finetune True -t 40
        python multi_match.py --date $date --model swin_reid --mode min --finetune True -t 40 -f ../32_33_AI_CUP_testdataset/AI_CUP_testdata/images/$date -l RE_RESULT/labels/$date --out MULTI_MATCH_RESULT

done

for date in "${date_list[@]}";
do
    python tools/datasets/AICUP_to_MOT15.py --AICUP_dir MULTI_MATCH_RESULT/labels/swin_reid_min_40/"$date" --MOT15_dir MOT15/MULTI_MATCH_RESULT/
done
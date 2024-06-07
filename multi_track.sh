date_list=(0902_130006_131041 0902_180000_181551 0903_125957_131610 0903_175958_181607 \
           0924_125953_132551 0924_175955_181603 0925_130000_131605 0925_175958_181604 \
           1001_130000_131559 1001_180000_181558 1002_130000_131600 1002_180000_181600 \
           1015_130000_131600 1015_180001_183846 1016_130000_131600 1016_180000_181600)

for date in "${date_list[@]}";
    do
        echo python multi_match.py --date $date --model swin_reid --mode min --finetune True -t 40
        python multi_match.py --date $date --model swin_reid --mode min --finetune True -t 40

done

for date in "${date_list[@]}";
do
    python tools/datasets/AICUP_to_MOT15.py --AICUP_dir RE_FINAL_result_v3/labels/swin_reid_min_40/"$date" --MOT15_dir MOT15/RE_firstSub_v3/
done
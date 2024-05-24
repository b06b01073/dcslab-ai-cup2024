
date_list=(0902_150000_151900 0903_150000_151900 0924_150000_151900 0925_150000_151900 1015_150000_151900 0902_190000_191900 0903_190000_191900 0924_190000_191900 0925_190000_191900 1015_190000_191900)

model_list=(swin_reid)


while getopts “m:s:e:t:?” argv
do
     case $argv in
         m)
            mode=$OPTARG
            ;;
         s)
            start=$OPTARG
            ;;
         e)
            end=$OPTARG
            ;;
         t)
            step=$OPTARG
            ;;
         ?)
            usage
            exit
            ;;
     esac
done


for((i=start; i<=end; i+=step))
do
    for model in "${model_list[@]}";
    do
        for date in "${date_list[@]}";
        do
            echo python multi_match.py --date $date --model $model --threshold $i --mode $mode --finetune True
            python multi_match.py --date $date --model $model --threshold $i --mode $mode --finetune True

            # Confirm whether the ground truth file exists.

            if ! [ -f MOT15/multi_cam_gt/$date.txt ];

            then
                echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir ../../LABEL/labels/$date --MOT15_dir MOT15/multi_cam_gt
                python tools/datasets/AICUP_to_MOT15.py --AICUP_dir ../../LABEL/labels/$date --MOT15_dir MOT15/multi_cam_gt
            fi

            echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir final_result/labels/"${model}_${mode}_${i}"/"$date" --MOT15_dir MOT15/multi_cam_ts/"${model}_${mode}_${i}"
            python tools/datasets/AICUP_to_MOT15.py --AICUP_dir final_result/labels/"${model}_${mode}_${i}"/"$date" --MOT15_dir MOT15/multi_cam_ts/"${model}_${mode}_${i}"
        done
        
        echo python tools/evaluate.py --gt_dir MOT15/multi_cam_gt/ --ts_dir MOT15/multi_cam_ts/"${model}_${mode}_${i}"
        python tools/evaluate.py --gt_dir MOT15/multi_cam_gt/ --ts_dir MOT15/multi_cam_ts/"${model}_${mode}_${i}"
    done
done
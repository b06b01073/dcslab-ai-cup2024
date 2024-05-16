#!/bin/bash

#default parameter
param="threshold"
start=50
end=60
step=10
# date_list=(0902_190000_191900 0903_150000_151900 0903_190000_191900 0924_150000_151900 \
# 0924_190000_191900 0925_150000_151900 0925_190000_191900 1015_150000_151900 1015_190000_191900)
date_list=(0902_150000_151900 0903_150000_151900 0924_150000_151900 0925_150000_151900 1015_150000_151900)
model_list=(resnet101_ibn_a resnext101_ibn_a densenet169_ibn_a se_resnet101_ibn_a swin_reid)
visualize=False
cam=3


while getopts “p:s:e:t:c:v:?” argv
do
     case $argv in
         p)
             param=$OPTARG
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
         v)
            visualize=$OPTARG
            ;;   
         c)
            cam=$OPTARG
            ;;
         ?)
             usage
             exit
             ;;
     esac
done
 


for((i=$start;i<=$end;i+=step))
do
    for date in "${date_list[@]}";
    do
        for model in "${model_list[@]}";
        do
            # Start tracking the cars on the monitor.
            echo python main.py -f IMAGE/$date -l LABEL/$date --$param $i --out dist_matrix/"${date}_${param}_${i}" --visualize $visualize --cam $cam --finetune True --model $model --output_ensemble True
            python main.py -f IMAGE/"${date}" -l LABEL/"${date}" --$param $i --out dist_matrix/"${date}_${param}_${i}" --visualize $visualize --cam $cam --finetune True --model $model --output_ensemble True
        done

        echo python ensemble.py -f IMAGE/$date -l LABEL/$date --$param $i --out aicup_ts/labels/"${date}_${param}_${i}" --visualize $visualize --cam $cam --dist_dir dist_matrix --finetune True
        python ensemble.py -f IMAGE/$date -l LABEL/$date --$param $i --out aicup_ts/labels/"${date}_${param}_${i}" --visualize $visualize --cam $cam --dist_dir dist_matrix --finetune True
        
        # Confirm whether the ground truth file exists.
        if ! [ -d MOT15/aicup_gt/$date ]
        then
            echo python parseAicup.py -s aicup_gt/labels/$date -l LABEL/$date/
            python parseAicup.py -s aicup_gt/labels/$date -l LABEL/$date/
            echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_gt/labels/$date --MOT15_dir MOT15/aicup_gt/$date
            python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_gt/labels/$date --MOT15_dir MOT15/aicup_gt/$date
        fi

        # Convert the results from AICUP format to MOT15 format.
        echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_ts/labels/"${date}_${param}_${i}"/$cam --MOT15_dir MOT15/aicup_ts/"${date}_${param}_${i}"
        python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_ts/labels/"${date}_${param}_${i}"/$cam --MOT15_dir MOT15/aicup_ts/"${date}_${param}_${i}"

        # Evaluate the result.
        echo python tools/evaluate.py --gt_dir MOT15/aicup_gt/$date --ts_dir MOT15/aicup_ts/"${date}_${param}_${i}" --mode single_cam --cam $cam
        python tools/evaluate.py --gt_dir MOT15/aicup_gt/$date --ts_dir MOT15/aicup_ts/"${date}_${param}_${i}" --mode single_cam --cam $cam
    done
done


for((i=$start;i<=$end;i+=step))
do
    for date in "${date_list[@]}";
    do
        echo
        python calculate_ave_performance.py -f ts_result/ -p "${param}" --start $start --end $end --step $step --cam $cam
    done
done
#!/bin/bash

#default parameter
param="threshold"
start=50
end=50
step=10
# date_list=(0902_190000_191900 0903_150000_151900 0903_190000_191900 0924_150000_151900 \
# 0924_190000_191900 0925_150000_151900 0925_190000_191900 1015_150000_151900 1015_190000_191900)
date_list=(0902_150000_151900 0903_150000_151900 0924_150000_151900 0925_150000_151900 1015_150000_151900)
model_list=(resnet101_ibn_a resnext101_ibn_a densenet169_ibn_a se_resnet101_ibn_a swin_reid)

visualize=False
time=m


while getopts “p:s:e:t:v:d:?” argv
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
         d)
            time=$OPTARG
            ;;
         ?)
             usage
             exit
             ;;
     esac
done


if [ $time == "m" ]
then
    date_list=(0902_150000_151900 0903_150000_151900 0924_150000_151900 0925_150000_151900 1015_150000_151900)
else
    date_list=(0902_190000_191900 0903_190000_191900 0924_190000_191900 0925_190000_191900 1015_190000_191900)
fi

for cam in {0..7}
do
    for((i=$start;i<=$end;i+=step))
    do
        for date in "${date_list[@]}";
        do
            for model in "${model_list[@]}";
            do

                if [ $param == "threshold" ]
                then
                    Label_from=""
                else
                    Label_from="YoloV8_"
                fi

                # Start tracking the cars on the monitor.
                echo python main.py -f IMAGE/$date -l LABEL/"$Label_from""$date" --$param $i --out aicup_ts/labels/"${date}_${param}_${i}" --visualize $visualize --cam $cam --finetune True --model $model
                python main.py -f IMAGE/$date -l LABEL/"$Label_from""$date" --$param $i --out aicup_ts/labels/"${date}_${param}_${i}" --visualize $visualize --cam $cam --finetune True --model $model

                # Confirm whether the ground truth file exists.
                if ! [ -d MOT15/aicup_gt/$date ]
                then
                    echo python parseAicup.py -s aicup_gt/labels/$date -l LABEL/$date/
                    python parseAicup.py -s aicup_gt/labels/$date -l LABEL/$date/
                    echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_gt/labels/$date --MOT15_dir MOT15/aicup_gt/$date
                    python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_gt/labels/$date --MOT15_dir MOT15/aicup_gt/$date
                fi

                # Convert the results from AICUP format to MOT15 format.
                echo python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_ts/labels/"${date}_${param}_${i}"/"$model"/$cam --MOT15_dir MOT15/aicup_ts/"${date}_${param}_${i}"/"$model"
                python tools/datasets/AICUP_to_MOT15.py --AICUP_dir aicup_ts/labels/"${date}_${param}_${i}"/"$model"/$cam --MOT15_dir MOT15/aicup_ts/"${date}_${param}_${i}"/"$model"

                # Evaluate the result.
                echo python tools/evaluate.py --gt_dir MOT15/aicup_gt/$date --ts_dir MOT15/aicup_ts/"${date}_${param}_${i}" --mode single_cam --cam $cam --model "$model"
                python tools/evaluate.py --gt_dir MOT15/aicup_gt/$date --ts_dir MOT15/aicup_ts/"${date}_${param}_${i}" --mode single_cam --cam $cam --model "$model"
            done
        done
    done
done

for cam in {0..7}
do
    for model in "${model_list[@]}"
    do
        echo python calculate_ave_performance.py -f ts_result -p "${param}" --start $start --end $end --step $step --cam $cam --model "$model" --time $time
        python calculate_ave_performance.py -f ts_result -p "${param}" --start $start --end $end --step $step --cam $cam --model "$model" --time $time
    done
done
python finetune_cnn.py --weights resnet101_ibn_a --save_dir weights/resnet_cent_final --backbone resnet
python finetune_cnn.py --weights resnext101_ibn_a --save_dir weights/resnext101_ibn_cent_final --backbone resnext
python finetune_swin.py --save_dir weights/swin_cent_final 


python finetune_cnn.py --weights se_resnet101_ibn_a --save_dir weights/se_resnet101_cent_final --backbone seresnet

python finetune_cnn.py --weights densenet169_ibn_a --save_dir weights/densenet_cent_final --backbone densenet


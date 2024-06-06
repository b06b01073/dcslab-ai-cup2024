# Readme

這個 subfolder 存放用來 fine-tune [pretrained model](https://github.com/b06b01073/veri776-pretrain/tree/cent) 的程式碼。

使用上需要傳入 xml 檔案路徑以及圖片路徑 see release (final.zip)

```
$ python contest_finetune/finetune_swin.py --save_dir tmp --dataset <path to the unzipped folder>
```

在上面提到的 release 當中也可以直接取用已經 fine-tune 好的檔案權重
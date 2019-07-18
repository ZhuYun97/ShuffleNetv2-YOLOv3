# ShufflNetv2-YOLOv3
The work is base on [YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch). I replace the backbone with ShuffleNet v2. And it is under testing now!
## Why this project
The computing complexity of darknet53 is costly. I want to speed up network computing. So I replace the backbone with ShuffleNet v2 which is a lightweight network in order to use the detector in mobile devices like smartphone. 
## Installation
### Environment
- pytorch >= 0.4.0
- python >= 3.6.0

### Get code
```git clone https://github.com/ZhuYun97/ShufflNetv2-YOLOv3.git```

### Download COCO dataset
```
cd ShufflNetv2-YOLOv3/data
bash get_coco_dataset.sh
```

## Training
### Download pretrained weights
- If you want to use ShuffleNetv2, you can downlaod the pretrained weights(emmmm, under training)
- if you want to use darknet, you just follow [the original author](https://github.com/BobLiu20/YOLOv3_PyTorch)

### Modify training parameters
1. Review config file training/params.py
2. Replace YOUR_WORKING_DIR to your working directory. Use for save model and tmp file.
3. Adjust your GPU device. see parallels.
4. Adjust other parameters.

### Start training
```
cd training
python training.py params.py
```

### Option: Visualizing training
```
#  please install tensorboard in first
python -m tensorboard.main --logdir=YOUR_WORKING_DIR   
```
## Evaluate
### Download pretrained weights
- If you want to use ShuffleNetv2, you can downlaod the pretrained weights(emmmm, under training)
- if you want to use darknet, you just follow [the original author](https://github.com/BobLiu20/YOLOv3_PyTorch)
Move downloaded file to wegihts folder in this project.

### Start evaluate
```
cd evaluate
python eval_coco.py params.py
```

## Quick test
### pretrained weights
Please download pretrained weights [in progress]() or use yourself checkpoint.
```
Start test
cd test
python test_images.py params.py
You can got result images in output folder.
```

## Measure FPS
pretrained weights
Please download pretrained weights [in progress]() or use yourself checkpoint.

## Start test
```
cd test
python test_fps.py params.py
```
## Results
Test in TitanX GPU with different input size and batch size.
Keep in mind this is a full test in YOLOv3. Not only backbone but also yolo layer and NMS.

## References
[YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch)

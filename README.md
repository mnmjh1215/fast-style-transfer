# Fast Style Transfer

Implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) in PyTorch

### Implementation details

I made some modifications.

1. Used instance normalization instead of batch normalization, because it [produces better images](https://arxiv.org/pdf/1607.08022)
2. Used VGG-19 with batch normalization instead of VGG-16

### How to run

To train model form scratch, use following command. 

(Note that content image path should be root directory that contains directory of images, because it was implemented using [torchvision ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). Also, style image path is path to image file, not directory.)

```
python main.py --content_image_path [content image directory path] --style_image_path [style image file path] --model_save_path [path to save model] --num_epochs [number of epochs]
```

To train model from existing checkpoint, add ```--model_path``` argument.

To generate new images using trained model, use following command.

```
python main.py --test --model_path [model file path] --test_image_path [test image directory path] --image_save_path [path to save generated images]
```

### Result

trained with following setting
```
--num_epochs 100 --lr 0.001 --batch_size 16 --style_weight 3e11 --content_weight 1e5
```

|   |![content_1](./images/content/472.jpg)|![content_2](./images/content/476.jpg)|
|---|---|---|
|![style](./images/style/mosaic.jpg)|![result_1](./images/generated/472.jpg)|![result_2](./images/generated/476.jpg)   |

### Future works

1. Support multiple style image. Current version only allows one style image.
2. Add total variation regularizer

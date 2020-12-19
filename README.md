# One Hundred Layers Tiramisu
PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326).

Tiramisu combines [DensetNet](https://arxiv.org/abs/1608.06993) and [U-Net](https://arxiv.org/abs/1505.04597) for high performance semantic segmentation. In this repository, we attempt to replicate the authors' results on the CamVid dataset.

This is a modified version for crack detection with focal loss. A tolerance margin is added to evaluate the crack detection.


## Setup
Install requirements.txt
```
pip install -r requirements.txt 
```

## Train

##### Dataset
Move the images of your dataset to `SegNet-Tutorial/CamVid/images`. Move your annotations to `SegNet-Tutorial/CamVid/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.png`. 

Dataset for crack detection:
CFD dataset: https://github.com/cuilimeng/CrackForest-dataset.git
The CFD dataset is saved under `SegNet-Tutorial/CamVid/CFD`.
The AigleRN dataset is saved under `SegNet-Tutorial/CamVid/AigleRN_RGB`.
##### Define Train and Validation Sets
Add paths to images that will be used as train and validation data respectively. In crack detection with the CFD dataset, the list is saved under `SegNet-Tutorial/CamVid/CFD/list`.
##### Command
train_focal.py  
Train the crack dataset with focal loss. The evaluation metrics are `precision, recall and F1` . Training information will be saved under the weight folder at the beginning of the training. Training loss and validation loss will be saved in `loss_train.csv` and  `validation.csv` under the `weights_folder` when all the training is finished.
```
$ train_focal.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--n_class N_CLASS] 
                [--pretrained_weights PRETRAINED_WEIGHTS]                
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--starts_epochs STARTS_EPOCHS]
                [--weights_folder WEIGHTS_FOLDER]
                [--path_folder PATH_FOLDER]
                [--alpha ALPHA][--gamma GAMMA]                
                [--train_list TRAIN_LIST][--valid_list VALID_LIST]
```
Note: More details could be added in the code.
1. For the evaluation metric, there is an `tolerance margin` with 5 pxiels. The change of the tolerance margin could be done in the code.
2. The normalizationn could be defined,given the mean and the standard deviation needs to be calculated first.
3. Random crop could also be defined.
4. The input channel could also be changed when training with depth map.
```
An example command at the terminal:
python3 train_focal.py --epochs 120 \
--n_class 2  --batch_size 2 \
--evaluation_interval 2 --checkpoint_interval 2 \
--gamma 2 --alpha 0.15 \
--weights_folder .weights/delete_later \
--train_list SegNet-Tutorial/CamVid/CFD/list/shallow_crack_with_aug_train.txt \
--valid_list SegNet-Tutorial/CamVid/CFD/list/shallow_crack_no_aug_val.txt 
```
train_normal.py  
The file provides training for normal object detection without focal loss and there is no tolerance margin for evaluation. And here another evaluation metric is provided: `pixel precision, mean class precision, mean IoU and weighted mIoU`. The definition of the evaluation metric could be found in `utils\training.py`. In this file, other evaluation function, such as `precision, recall and F1` are also provided.


## Evaluation and detection
valuation_weight.py evaluate a dataset with weight files. The output are `precision, recall and F1` for the weight. The result is saved in csv file under the weight folder.
```
$ evaluation_weight.py [-h] [--batch_size BATCH_SIZE][--n_class N_CLASS] 
                [--path_folder PATH_FOLDER]  
                [--predict_list PREDICT_LIST] 
                [--start_epoch START_EPOCH][--end_epoch END_EPOCH]
                [--step STEP][--filename FILENAME]
                [--tolerance TOLERANCE]
```

```
An example command at the terminal:
python3 evaluation_weight.py --path_folder .weights/alpha015_save_weights \
--start_epoch 94 \
--end_epoch 96 \
--step 2 \
--predict_list SegNet-Tutorial/CamVid/CFD/list/shallow_crack_no_aug_val.txt \
--filename val.csv
```
prediction_image.py  
The file generates prediction results on the image with weight file. 

```
An example command at the terminal:
python3 prediction_image.py \
--weights_file_folder .weights/alpha015_save_weights/ \
--weight_filename weights-94.pth \
--result_folder output/prediction_result/ \
--test_list SegNet-Tutorial/CamVid/CFD/list/shallow_crack_no_aug_val.txt 
```


## Architecture

Tiramisu adopts the UNet design with downsampling, bottleneck, and upsampling paths and skip connections. It replaces convolution and max pooling layers with Dense blocks from the DenseNet architecture. Dense blocks contain residual connections like in ResNet except they concatenate, rather than sum, prior feature maps.



## References and Links

* [Project Thread](http://forums.fast.ai/t/one-hundred-layers-tiramisu/2266)
* [Author's Implementation](https://github.com/SimJeg/FC-DenseNet)
* https://github.com/bamos/densenet.pytorch
* https://github.com/liuzhuang13/DenseNet

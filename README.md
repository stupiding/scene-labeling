#Notice
This code is still under test and modification. If you are interested in the model, you may also visit (https://github.com/stupiding/rcnn), which implements an essentially same model for image classification.

The project aims to providing the Torch solution for the paper [Convolutional Neural Networks with Intra-Layer Recurrent Connections for Scene Labeling](http://xlhu.cn/papers/Liang15-nips.pdf). The code is modified from [facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

#Requirements

1. A GPU machine with Torch and its cudnn bindings. See [Installing Torch](http://torch.ch/docs/getting-started.html#_).

2. Download the siftflow dataset, tranform it to t7 format, and put it to scene-labeling/data/siftflow.

#How to use
Run main.lua with options to train network models.

An example is:

`CUDA_VISIBLE_DEVICES=0,1 th main.lua -dataset siftflow -model rcl3 -nGPU 2 -nThreads 4 -lr 0.1 -nChunks 100 -batchSize 64`

To see all options and their default value, run:

`th main.lua -help`

#Code introduction

1. main.lua: Overall procedure to run the code.

2. dataset.lua: Prepare mini-batchs from specified datasets, including possible data augmentation.

3. data.lua: Initiate the dataset and setup multi-thread data loaders.

4. model.lua: Initiate the network models. Model files are placed in rcnn/models/.

5. train.lua: Train and test network models.

6. parse.lua: Parse the input options.

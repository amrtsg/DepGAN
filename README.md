# Depth GAN (DepGAN): Leveraging Depth Maps for Handling Occlusions and Transparency in Image Composition

![alt text](https://github.com/amrtsg/DepGAN/blob/master/misc/results.jpg?raw=true)

## Code Setup

**The code was developed and tested on:** <br>
```
Ubuntu 22.04 / WSL
Python 3.10.13
Tensorflow 2.15.0
Cuda 12.1
Ryzen 7 3800x
Nvidia RTX 3090
```

## Dataset

We have provided Shapenet's basket-bottle dataset in the ```data/bottle``` folder. Including the depth estimations for each background image in the ```depth``` folder.
To create your own dataset, we recommend using MiDaS to estimate the depth images.

## Environment

*Please make sure you have Cuda 12.1 installed on your machine.*
```
> conda create -n depgan python=3.10.13 -y && conda activate depgan
> pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
> pip install -U tensorflow[and-cuda]==2.15.0
> pip install -r requirements.txt
```
## Running Code

We have put everything together into 2 jupyter notebooks, train.ipynb and predict.ipynb

To train depgan, run ```train.ipynb```. This trains and saves the model at every epoch.
NOTE: You can expect the training run to take 10 minutes for 100 epochs on a 3090, we recommend 150 epochs.

To test depgan, run ```predict.ipynb```, this will generate the predictions and output the results in a plot.
NOTE: Make sure to pick the correct saved model from the logs/model folder, this is found in the line 
```
model = load_model('logs/cgan/models/g_model_epochXXX.h5')
```

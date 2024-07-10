## Depth GAN (DepGAN): Leveraging Depth Maps for Handling Occlusions and Transparency in Image Composition

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

## Datasets

We have provided Shapenet's basket-bottle dataset in the ```data/bottle``` folder. Including the depth estimations for each background image in the ```depth``` folder.
To create your own dataset, we recommend using MiDaS to estimate the depth images.

Dataset | Size |
--- | --- |
[Bottle-Basket](https://drive.google.com/file/d/1Ef2QiSi_meIw14btTd0qq47Gp0qIYnVb/view?usp=drive_link) | 13.1 MB |
[Chair-Desk](https://drive.google.com/file/d/1UKpqTcjsKd1VR0RI8JWtNwCGYUojcpEm/view?usp=drive_link) | 81.0 MB |
Glasses-Face | NA |
Aerial | NA |
Real Car | NA |

## Pretrained Models

Dataset | Model | Size |
--- | --- | --- |
Bottle-Basket | [bottle_basket_75](https://drive.google.com/file/d/1Jgk2CX2lxvhVznXc9CkR7TXWcq02bzAd/view?usp=drive_link) | 239 MB |
Chair-Desk | [chair_desk_164](https://drive.google.com/file/d/1qmmy5vO-TrVjOWNqE_C3A8xAHo11X34c/view?usp=drive_link) | 239 MB |
Glasses-Face | Coming soon... | NA |
Aerial | Coming soon... | NA |
Real Car | Coming soon... | NA |

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

To train depgan, run ```train.ipynb```. This trains and saves the model at every epoch in the ```logs/models```. A plot with generated samples will be saved at ```logs/plots```.
NOTE: You can expect the training run to take 10 minutes for 100 epochs on a 3090, we recommend 150 epochs.

To test depgan, run ```predict.ipynb```, this will generate the predictions and save the output to ```logs/generated/{DATASET}```.

All variables can be found and set in ```config.py```. These include config variables to control GUI, dataset paths, and model hyperparameters. Please make sure you have the correct configuration for your train/test run.

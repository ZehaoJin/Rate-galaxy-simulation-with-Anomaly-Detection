<<<<<<< HEAD
# UPDATE: 
This repo is no longer maintained. [GANomaly](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/ganomaly) implementation has been added to [anomalib](https://github.com/openvinotoolkit/anomalib), the largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.

# GANomaly

This repository contains PyTorch implementation of the following paper: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [[1]](#reference)

##  1. Table of Contents
- [GANomaly](#ganomaly)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Experiment](#experiment)
    - [Training](#training)
        - [Training on MNIST](#training-on-mnist)
        - [Training on CIFAR10](#training-on-cifar10)
        - [Train on Custom Dataset](#train-on-custom-dataset)
    - [Citing GANomaly](#citing-ganomaly)
    - [Reference](#reference)
    

## 2. Installation
1. First clone the repository
   ```
   git clone https://github.com/samet-akcay/ganomaly.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n ganomaly python=3.7
    ```
3. Activate the virtual environment.
    ```
    conda activate ganomaly
    ```
3. Install the dependencies.
   ```
   conda install -c intel mkl_fft
   pip install --user --requirement requirements.txt
   ```

## 3. Experiment
To replicate the results in the paper for MNIST and CIFAR10  datasets, run the following commands:

``` shell
# MNIST
sh experiments/run_mnist.sh

# CIFAR
sh experiments/run_cifar.sh # CIFAR10
```

## 4. Training
To list the arguments, run the following command:
```
python train.py -h
```

### 4.1. Training on MNIST
To train the model on MNIST dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset mnist                         \
    --niter <number-of-epochs>              \
    --abnormal_class <0,1,2,3,4,5,6,7,8,9>  \
    --display                               # optional if you want to visualize     
```

### 4.2. Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset cifar10                                                   \
    --niter <number-of-epochs>                                          \
    --abnormal_class                                                    \
        <plane, car, bird, cat, deer, dog, frog, horse, ship, truck>    \
    --display                       # optional if you want to visualize        
```

### 4.3. Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Then model training is the same as training MNIST or CIFAR10 datasets explained above.

```
python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>      \
    --display                       # optional if you want to visualize
```

For more training options, run `python train.py -h`.

## 5. Citing GANomaly
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@inproceedings{akcay2018ganomaly,
  title={Ganomaly: Semi-supervised anomaly detection via adversarial training},
  author={Akcay, Samet and Atapour-Abarghouei, Amir and Breckon, Toby P},
  booktitle={Asian Conference on Computer Vision},
  pages={622--637},
  year={2018},
  organization={Springer}
}
```

## 6. Reference
[1]  Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
=======
# Rate-galaxy-simulation-with-Anomaly-Detection

## [Paper in prep](https://www.overleaf.com/read/ddzwssghsqhy)

## [Presentation Slides](GANomaly_20230321_Zehao.pptx)

## 1. Overview
Train a neural network to reconstruct (original image -> feature space represetation of image -> reconstruct image back) SDSS galaxy images, with only SDSS images as training set. After trained, let the neural network reconstruct galaxy simulation images. The places where the reconstruction fails is the anomaly. The distance in latent space between original images and reconstructed image serves as Anomaly score $\mathcal{A}$. Lower $\mathcal{A}$ means the simulation is closer to SDSS observations.

Here as an example we treat SDSS galaxy RGB ($i$ - $r$ - $g$ band) images as normal set, and detect anomalies in NIHAO simulated galaxies (mock observed in SDSS $i$ - $r$ - $g$ band). This model can be used on any other galaxy simulations without any re-train, can be extended to other surveys other than SDSS or other maps (velocity maps) with re-train.


## 2. Network Architecture - GANomaly ([Samet Akcay et al.](https://arxiv.org/abs/1805.06725))
![](plots/ganomaly.png)

## 3. Gallery  (original galaxy | reconstructed galaxy | residual)
### 3.1. SDSS test set
![](plots/SDSSlow1.png)![](plots/SDSSlow2.png)![](plots/SDSSlow3.png)
![](plots/SDSSmid1.png)![](plots/SDSSmid2.png)![](plots/SDSSmid3.png)
![](plots/SDSShigh1.png)![](plots/SDSShigh2.png)![](plots/SDSShigh3.png)
### 3.2. NIHAO simulated galaxies
![](plots/NoAGNlow.png)![](plots/NoAGNmid.png)![](plots/NoAGNhigh.png)
![](plots/AGNlow.png)![](plots/AGNmid.png)![](plots/AGNhigh.png)
### 3.3. Sanity Check
![](plots/apple.png)![](plots/SDSSabnormal1.png)![](plots/SDSSabnormal2.png)

## 4. Anomaly Score Statistics
### 4.1. General
![](plots/SDSSnihao.png)
![](plots/grandscore.png)
### 4.2. NIHAO AGN vs NIHAO NoAGN
![](plots/stellarmass.png)
![](plots/NIHAONoAGNAGN.png)
### 4.3. NIHAO n10 vs NIHAO n80 (Tune different star formation threshold parameter)
![](plots/n80score.png)
### 4.4. NIHAO UD vs NIHAO UHD (lower vs. higher simulation resolution)
![](plots/UHDscore.png)
### 4.5. $\mathcal{A}$ vs scaling relation properties
![](plots/massA.png)
![](plots/SFR100A.png)
![](plots/SFR500A.png)
![](plots/ZA.png)
![](plots/ageA.png)
![](plots/BVA.png)
### 4.6. $\mathcal{A}$ vs scaling relations
![](plots/nik1.png)
![](plots/nik2.png)
![](plots/nik3.png)
![](plots/nik4.png)
![](plots/nik5.png)
>>>>>>> ab409c0433a9c2389be9cf1fd9ec9bbe4fbfa872

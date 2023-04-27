# Rate-galaxy-simulation-with-Anomaly-Detection

## [Paper in prep](https://www.overleaf.com/read/ddzwssghsqhy)

## [Presentation Slides](GANomaly_20230321_Zehao.pptx)

## [Glance of this project](#glance-of-this-project-1)

## [Use this project](#use-this-project-1)

# Glance of this project
Code to produce all the plots in the paper/in below is located at [paper_plot.ipynb](paper_plot.ipynb)

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

# Use this project

Installation[#1-installation]
 

## 1. Installation
First, clone this repository to your machine

    git clone git@github.com:ZehaoJin/Rate-galaxy-simulation-with-Anomaly-Detection.git
    
or

    git clone https://github.com/ZehaoJin/Rate-galaxy-simulation-with-Anomaly-Detection.git

make sure your machine has a GPU

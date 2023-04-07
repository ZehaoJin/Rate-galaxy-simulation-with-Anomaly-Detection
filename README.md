# Rate-galaxy-simulation-with-Anomaly-Detection

## [Paper in prep](https://www.overleaf.com/read/ddzwssghsqhy)

## [Presentation Slides](GANomaly_20230321_Zehao.pptx)

## Overview
Train a neural network to reconstruct (original image -> feature space represetation of image -> reconstruct image back) SDSS galaxy images, with only SDSS images as training set. After trained, let the neural network reconstruct galaxy simulation images. The places where the reconstruction fails is the anomaly. The distance in latent space between original images and reconstructed image serves as Anomaly score $\mathcal{A}$. Lower $\mathcal{A}$ means the simulation is closer to SDSS observations.

Here as an example we treat SDSS galaxy RGB ($i$ - $r$ - $g$ band) images as normal set, and detect anomalies in NIHAO simulated galaxies (mock observed in SDSS $i$ - $r$ - $g$ band). This model can be used on any other galaxy simulations without any re-train, can be extended to other surveys other than SDSS or other maps (velocity maps) with re-train.


#### Network Architecture - GANomaly ([Samet Akcay et al.](https://arxiv.org/abs/1805.06725))
![](plots/ganomaly.png)

#### Gallery  (original galaxy | reconstructed galaxy | residual)
##### SDSS test set
![](plots/SDSSlow1.png)![](plots/SDSSlow2.png)![](plots/SDSSlow3.png)
![](plots/SDSSmid1.png)![](plots/SDSSmid2.png)![](plots/SDSSmid3.png)
![](plots/SDSShigh1.png)![](plots/SDSShigh2.png)![](plots/SDSShigh3.png)
##### NIHAO simulated galaxies
![](plots/NoAGNlow.png)![](plots/NoAGNmid.png)![](plots/NoAGNhigh.png)
![](plots/AGNlow.png)![](plots/AGNmid.png)![](plots/AGNhigh.png)
##### Sanity Check
![](plots/apple.png)![](plots/SDSSabnormal1.png)![](plots/SDSSabnormal2.png)

#### Anomaly Score Statistics
##### General
![](plots/SDSSnihao.png)
![](plots/grandscore.png)
##### NIHAO AGN vs NIHAO NoAGN
![](plots/stellarmass.png)
![](plots/NIHAONoAGNAGN.png)
##### NIHAO n10 vs NIHAO n80 (Tune different star formation threshold parameter)
![](plots/n80score.png)
##### NIHAO UD vs NIHAO UHD (lower vs. higher simulation resolution)
![](plots/UHDscore.png)
##### $\mathcal{A}$ vs scaling relation properties
![](plots/massA.png)
![](plots/SFR100A.png)
![](plots/SFR500A.png)
![](plots/ZA.png)
![](plots/ageA.png)
![](plots/BVA.png)
##### $\mathcal{A}$ vs scaling relations
![](plots/nik1.png)
![](plots/nik2.png)
![](plots/nik3.png)
![](plots/nik4.png)
![](plots/nik5.png)

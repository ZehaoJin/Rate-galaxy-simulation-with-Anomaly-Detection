# Rate-galaxy-simulation-with-Anomaly-Detection

### [Paper: Quantitatively rating galaxy simulations against real observations with anomaly detection](https://academic.oup.com/mnras/article/529/4/3536/7612260)
### [Installation](#1-installation)
### [Use the SDSS-trained model described in the paper](#2-use-the-sdss-trained-model-described-in-the-paper)
### [Train your own new model with a new dataset](#3-train-your-own-new-model-with-a-new-dataset)
### [Reproduce plots in the paper](#4-reproduce-plots-in-the-paper)
### [Download SDSS galaxy image dataset](#5-download-sdss-galaxy-image-dataset)
### [Feature/latent space exploration](#6-featurelatent-space-exploration)
### [Cite this work](#7-cite-this-work)


## 1. Installation
### 1.1. clone this repository to your machine (make sure your machine has a GPU)

    git clone git@github.com:ZehaoJin/Rate-galaxy-simulation-with-Anomaly-Detection.git

or

    git clone https://github.com/ZehaoJin/Rate-galaxy-simulation-with-Anomaly-Detection.git

### 1.2. Install dependencies

- If you already have many Machine Learning/Astro packages installed, maybe (although not recommanded) you can skip this step, and install needed packages when import error appears (`conda install <package-name>`). Some code might need a bit more dependencies than others. Typical packages you will need are pytorch, scikit-learn, astropy, visdom, tqdm, pandas, numpy, matplotlib, etc

- (Recommanded) Create a virtual environment via conda:

      conda create --name ganomaly

- Activate the virtual environment:

      conda activate ganomaly

- Install the dependencies:

      conda install -c intel mkl_fft
      pip install --user --requirement requirements.txt


## 2. Use the SDSS-trained model described in the paper

- You can use this already trained model to rate ANY simulations -- it's never limted to NIHAO simulations.
- Prepare your SDSS $i-r-g$ band mock observed galaxy images. $3\times 64\times 64$ in size, $i-r-g$ to $R-G-B$.
- Put images you want to test  under folder `data/test/test/1.abnormal/`
- Run jupyter notebook [test.ipynb](test.ipynb), simply run all cells to get anomaly scores and reconstruction plots.
- Or, in terminal, run `python test.py`. The result anomaly score will be saved at output/ganomaly/test/test/anomaly_scores.csv . See the desciptions in [test.py](test.py) for more options.


## 3. Train your own new model with a new dataset

- You shall train GANomaly on different telescopes (other than SDSS), different bands (other than SDSS $i-r-g$), different image size (other than $3\times 64\times 64$), and different maps (velocity maps, density maps, etc), to spot their anomalies.
- Prepare your "normal" set of data, usually from observations. Note they do not strictly have to be all perfectly "normal", but make sure most of them are "normal".
- Copy your data to  `data/<your-dataset-name>` directory, with the following directory & file structure. Note that it's perfectly okay to leave `test/0.normal` and `test/1.abnormal` empty during training.

      <your-dataset-name>
       ├── test
       │   ├── 0.normal
       │   │   └── normal_img_0.png
       │   │   └── normal_img_1.png
       │   │   ...
       │   │   └── normal_img_n.png
       │   ├── 1.abnormal
       │   │   └── abnormal_img_0.png
       │   │   └── abnormal_img_1.png
       │   │   ...
       │   │   └── abnormal_img_m.png
       ├── train
       │   ├── 0.normal
       │   │   └── normal_img_0.png
       │   │   └── normal_img_1.png
       │   │   ...
       │   │   └── normal_img_t.png

- In a seperate terminal, run `visdom`, which allows you to visualize the whole training progress.

      python3 -m visdom.server

- Alternatively, you can run visdom in a "virval terminal" with `screen`:

      screen -S <your-session-name>
      python3 -m visdom.server

  And `Ctrl+A`,then press `D` to detach that screen session, and leave visdom run in the background.

  You can re-attach to that session by `screen -r <your-session-name>`, check all sessions by `screen -ls`, kill the current attached sesssion by `Ctrl+A`, then press `K`

- Start training:

      python train.py                        \
         --dataset <your-dataset-name>       \
         --isize <image-size>                \
         --nc <input-image-channels>         \
         --nz <size-of-the-latent-z-vector>  \
         --niter <number-of-epochs>          \

  For more training options, run `python train.py -h`

- Go to your favourite browser, go to address: `localhost:8097` to monitor your training.

- You will find your trained weights, losslog, options log, etc. in `output/`

- To test your trained model, again use [test.ipynb](test.ipynb) or [test.py](test.py). Remember to change the path of weights to your newly trained one, and change options parameters to match the ones you used during training.

- It is almost equivalent to use this repository or [the original GANomaly repository by Samet Akcay et al.](https://github.com/samet-akcay/ganomaly) to train.


## 4. Reproduce plots in the paper
Code to produce all the plots in the paper/in below is located at [paper_plot_0211.ipynb](paper_plot_0211.ipynb)


## 5. Download SDSS galaxy image dataset

- To download 670,722 SDSS galaxy image dataset used in this paper. The images are from SDSS image cutout tool, galaxy coordinates are from galaxy catalog by [Meert et al. (2014)](https://academic.oup.com/mnras/article/446/4/3943/2892484)
- Run this [python script](SDSS_cutouts/download_cutouts.py) `python SDSS_cutouts/download_cutouts.py` to download the cutout images to `cutout` folder.
- Tips: It takes long to download all the images... maybe you want to download it in the background via things like `screen`
- Run this [python script](SDSS_cutouts/split_train_test.py) `python SDSS_cutouts/split_train_test.py` will create a training/test dataset for you.


## 6. Feature/latent space exploration

[explore_features.ipynb](explore_features.ipynb)


## 7. Cite this work
If you use this repository or would like to refer the paper, please use the following BibTeX entry:

    @article{10.1093/mnras/stae552,
        author = {Jin, Zehao and Macciò, Andrea V and Faucher, Nicholas and Pasquato, Mario and Buck, Tobias and Dixon, Keri L and Arora, Nikhil and Blank, Marvin and Vulanovic, Pavle},
        title = "{Quantitatively rating galaxy simulations against real observations with anomaly detection}",
        journal = {Monthly Notices of the Royal Astronomical Society},
        volume = {529},
        number = {4},
        pages = {3536-3549},
        year = {2024},
        month = {02},
        abstract = "{Cosmological galaxy formation simulations are powerful tools to understand the complex processes that govern the formation and evolution of galaxies. However, evaluating the realism of these simulations remains a challenge. The two common approaches for evaluating galaxy simulations is either through scaling relations based on a few key physical galaxy properties, or through a set of pre-defined morphological parameters based on galaxy images. This paper proposes a novel image-based method for evaluating the quality of galaxy simulations using unsupervised deep learning anomaly detection techniques. By comparing full galaxy images, our approach can identify and quantify discrepancies between simulated and observed galaxies. As a demonstration, we apply this method to SDSS imaging and NIHAO simulations with different physics models, parameters, and resolution. We further compare the metric of our method to scaling relations as well as morphological parameters. We show that anomaly detection is able to capture similarities and differences between real and simulated objects that scaling relations and morphological parameters are unable to cover, thus indeed providing a new point of view to validate and calibrate cosmological simulations against observed data.}",
        issn = {0035-8711},
        doi = {10.1093/mnras/stae552},
        url = {https://doi.org/10.1093/mnras/stae552},
        eprint = {https://academic.oup.com/mnras/article-pdf/529/4/3536/57110995/stae552.pdf},
    }



It is always nice to cite the original GANomaly paper too:

    @inproceedings{akcay2018ganomaly,
      title={Ganomaly: Semi-supervised anomaly detection via adversarial training},
      author={Akcay, Samet and Atapour-Abarghouei, Amir and Breckon, Toby P},
      booktitle={Asian Conference on Computer Vision},
      pages={622--637},
      year={2018},
      organization={Springer}
    }

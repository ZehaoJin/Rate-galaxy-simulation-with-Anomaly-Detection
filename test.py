"""
Put images you want to test in data/test/1.abnormal/, or data/test/0.normal/
images should be 3*64*64  if you are using the SDSS-trained model descripted in the paper

result anomaly score is saved at output/output/test/anomaly_scores.csv
result images is saved at output/output/test/images/   (only if you uncomment line 128 & 132)


Test the SDSS-trained model described in the paper:
    In terminal, run:
        python test.py


Test your own trained model:
    replace the weights path with your new one
        i.e. change path = "./output/ganomaly/SDSS/train/weights/netG_43.pth" to path = "your path"  (at line 84)
    In termial, run:
        python test.py --isize 64  --nc 3 --nz 128

        ## adjust isize, nc, nz to match your training
"""

from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

from collections import OrderedDict
import glob,os
import shutil
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate
import pandas as pd



def show(imgs,score,name):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fix.suptitle(name+' $\mathcal{A}$ = '+str(score),y=0.77)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
##
def test():

    ##
    # ARGUMENTS
    opt = Options().parse()
    opt.dataset='test'
    opt.batchsize=1     #batch size, set to 1 when testing
    opt.isTrain = False
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # test
    with torch.no_grad():
        # Load the weights of netg and netd.
        if model.opt.load_weights:
            path = "./output/ganomaly/SDSS/train/weights/netG_43.pth"
            pretrained_dict = torch.load(path)['state_dict']

            try:
                model.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")
            print('   Loaded weights.')

        model.opt.phase = 'test'

        #model.netg.eval()
        # note: why not use model.netg.eval(), but only torch.no_grad() ? (see: https://github.com/samet-akcay/ganomaly/issues/83)
        # "This has come up before with the associated skip-ganomaly repo (https://github.com/samet-akcay/skip-ganomaly)
        #  I asked Samet Akcay, the first author [I am also a co-author on the GANomaly paper but my familiarity with the code-base
        # is essentially zero] and he said "..it’s not a bug. During the DCGAN era, it was common not to put the model on eval mode
        # for the stability. What I did instead, is to use batch norm, and batch size of 1 during the test so the performance is not
        # affected.
        # If you add .eval() it significantly reduces the AUC performance against this batch size of 1 approach
        # - ourselves and others have found this. AFIAK, there is no dropout in use in this approach." -- tobybreckon

        loss_con=nn.L1Loss(reduction='none')
        ascore_array=np.zeros(len(model.dataloader['test']))

        for i, data in tqdm(enumerate(model.dataloader['test'], 0),total=len(model.dataloader['test'])):
            model.set_input(data)
            model.fake, latent_i, latent_o = model.netg(model.input)

            error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)

            ascore = error.reshape(error.size(0))

            ascore_array[i]=ascore

            #original_feature=latent_i.cpu().detach().numpy()
            #original_feature=original_feature.reshape((Args.batchsize,Args.nz))

            real, fake, _ = model.get_current_images()
            loss_img=loss_con(model.fake,model.input).data
            grid=make_grid([real[0],fake[0],loss_img[0]],normalize=True)
            name=model.dataloader['test'].dataset.samples[i][0]

            ### uncomment if you want to plot the images,
            ### could be slow if there are too many, or ran into display issues if you running remotely
            #show(grid,ascore_array[i],name)

            ### uncomment if you want to further save the images.
            ### the output image is saved at ./output/output/test/image/ folder
            #plt.savefig(opt.outf+'/output/test/images/'+str(i)+'.png')

        df=pd.DataFrame(model.dataloader['test'].dataset.samples,columns=['filename','ascore'])
        df['ascore']=ascore_array
        df.to_csv(opt.outf+'/output/test/anomaly_scores.csv',index='False')


if __name__ == '__main__':
    test()

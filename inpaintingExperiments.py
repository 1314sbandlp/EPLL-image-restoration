import os
import numpy as np
from utils import rgb2gray
from skimage.io import imread,imsave
from constructGMM import GMM_from_learned_model
from aprxMAPGMM import aprxMAPGMM
from EPLLhalfQuadraticSplit import EPLL_half_quadratic_split
from utils import GMM_log_p,get_logger,im2col
from patchDCTGG import patch_DC_TGG
from skimage.color import rgb2ycbcr, ycbcr2rgb

if __name__=="__main__":
    save_path = "inpainting_result"

    # create save dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    patch_size=8
    I = (imread('new.jpg'))/255;
    mask = (imread('new_mask.png'))/255.;

    if len(I.shape)==3:
        I = rgb2ycbcr(I)/255.;

    #find which patches are occluded for faster performance
    noiseI=np.copy(I)
    mask_index=mask>0
    tt=noiseI[:,:,0]
    tt[mask_index]=-999.0
    ttt,_,_=im2col(tt,patch_size)
    exclude_list=[np.sum(ttt[i]==-999.0)!=ttt.shape[1] for i in range(ttt.shape[0])]

    GMM_path = "GSModel_8x8_200_2M_noDC_zeromean"
    GS = GMM_from_learned_model(GMM_path)
    # change prior if needed
    def denoise_GMM_prior(z, patch_size, noiseSD, imsize):
        return aprxMAPGMM(z, patch_size, noiseSD, imsize, GS,exclude_list)


    prior = denoise_GMM_prior
    # change log function if needed
    log_fuction = lambda Z: GMM_log_p(Z, GS)

    for i in range(I.shape[-1]):
        tt = np.copy(I[:,:,i])
        tt[mask_index] = 0
        noiseI[:,:,i]=tt


    # inpainting
    lamb = 1000000*np.ones([I.shape[0],I.shape[1]]);
    lamb[mask_index]=0;
    cleanI = np.zeros(I.shape);

    import matplotlib.pyplot as plt

    for i in range(I.shape[-1]):
        cleanI[:,:,i],_,_,_ = EPLL_half_quadratic_split(noiseI[:,:,i],lamb,patch_size,1,10*[1,2,16,128,512],prior,I[:,:,i])


    if len(I.shape)==3:
        I = ycbcr2rgb(I*255.)
        cleanI=ycbcr2rgb(cleanI*255.)
        noiseI=ycbcr2rgb(noiseI*255.)


    imsave(f"{save_path}/inpainting_original.jpg",I)
    imsave(f"{save_path}/inpainting_noise.jpg",noiseI)
    imsave(f"{save_path}/inpainting_clean.jpg",cleanI)
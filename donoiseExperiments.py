'''
Test multiple images in denoising
'''

import os
import numpy as np
from utils import rgb2gray
from skimage.io import imread,imsave
from constructGMM import GMM_from_learned_model
from aprxMAPGMM import aprxMAPGMM
from EPLLhalfQuadraticSplit import EPLL_half_quadratic_split
from utils import GMM_log_p,get_logger
from patchDCTGG import patch_DC_TGG


if __name__=="__main__":
    np.random.seed(234)
    #define argments
    test_data_path="test_dataset"
    save_path="denoise_result_ICA_betaestimate"
    patch_size=8
    sigma=25
    noiseSD = sigma/255;
    GMM_path="GSModel_8x8_200_2M_noDC_zeromean"
    ICA_path="ICAModel"
    GS=GMM_from_learned_model(GMM_path)
    lamb=patch_size**2/noiseSD**2
    T=1
    #use estimated beta, add -1 in the begining
    betas=-1*(1/noiseSD**2)*np.array([1,4,8,16,32])

    #create save dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #create and save parameters to log file
    log = get_logger(save_path,"logger")
    log.info(f"sigma: {sigma}")
    log.info(f"noiseSD: {noiseSD}")
    log.info(f"lambda: {lamb}")
    log.info(f"T : {T}")
    log.info(f"betas: {betas}")


    #change prior if needed
    def denoise_GMM_prior(z,patch_size,noiseSD,imsize):
        return aprxMAPGMM(z,patch_size,noiseSD,imsize,GS)
    prior=denoise_GMM_prior
    #change log function if needed
    log_fuction=lambda Z:GMM_log_p(Z,GS)

    """
    #ICA prior 
    W=np.loadtxt(f"{ICA_path}/ICA_W.txt",delimiter=",")
    E=np.loadtxt(f"{ICA_path}/ICA_E.txt",delimiter=",")

    W=np.matmul(W,E)
    invW=np.linalg.pinv(W)

    def denoise_ICA_prior(z,patch_size,noiseSD,imsize):
        return patch_DC_TGG(z,patch_size,noiseSD,imsize,W,invW)
    prior=denoise_ICA_prior
    log_fuction=None
"""

    #load image in test file
    pathDir = os.listdir(test_data_path)
    num_img=len(pathDir)
    #for result saving
    PSNR_list=np.zeros([num_img,T*len(betas)])
    PSNR_I1_list=np.zeros([num_img,T*len(betas)])
    cost_list=np.zeros([num_img,T*len(betas)])

    #run and save result to save_path
    for i,file in enumerate(pathDir):
        log.info(f"test in image {file}")
        I=np.float32(rgb2gray(imread(f"{test_data_path}/{file}")))/255.
        noise_image =I + noiseSD * np.random.normal(0, 1, I.shape[0] * I.shape[1]).reshape(I.shape)
        clean_iamge,PSNR,PSNR_I1,cost=EPLL_half_quadratic_split(noise_image,lamb,patch_size,T,betas,prior,I,log_fuction,log)
        imsave(f"{save_path}/{file.split('.')[0]}_original_image.jpg", I)
        imsave(f"{save_path}/{file.split('.')[0]}_noise_image.jpg", noise_image)
        imsave(f"{save_path}/{file.split('.')[0]}_denoise_result.jpg", clean_iamge)
        PSNR_list[i]=np.array(PSNR)
        PSNR_I1_list[i]=np.array(PSNR_I1)
        if(len(cost)>0):
            cost_list[i]=np.array(cost)
        log.info(f"test in image {file} complete")

    np.savez(f"{save_path}/result.npz",PSNR=PSNR_list,PSNR_I1=PSNR_I1_list,cost=cost_list)
    log.info(f"mean PSNR is : {np.mean(PSNR_list[:,-1])} ")
    log.info(f"mean PSNR_I1 is : {np.mean(PSNR_I1_list[:,-1])} ")





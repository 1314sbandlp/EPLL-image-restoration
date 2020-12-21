'''
Test one sample image in denoising
'''
import numpy as np
from utils import rgb2gray
from skimage.io import imread,imsave
from constructGMM import GMM_from_learned_model
from aprxMAPGMM import aprxMAPGMM
from EPLLhalfQuadraticSplit import EPLL_half_quadratic_split
from utils import GMM_log_p
import os


if __name__=="__main__":
    save_path="denoise_result"

    # create save dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.random.seed(234)
    #load image
    img=np.float32(rgb2gray(imread('160068.jpg')))/255.
    patch_size=8
    #add noise
    noiseSD = 25/255;
    noise_image = img + noiseSD*np.random.normal(0,1,img.shape[0]*img.shape[1]).reshape(img.shape)
    #noise_image=np.loadtxt("noise_image.txt",delimiter=",")

    GMM_path="GSModel_8x8_200_2M_noDC_zeromean"
    GS=GMM_from_learned_model(GMM_path)

    def denoise_GMM_prior(z,patch_size,noiseSD,imsize):
        return aprxMAPGMM(z,patch_size,noiseSD,imsize,GS)

    lamb=patch_size**2/noiseSD**2
    T=1
    betas=(1/noiseSD**2)*np.array([-1,-4,-8,-16,-32])
    prior=denoise_GMM_prior
    I=img
    log_fuction=lambda Z:GMM_log_p(Z,GS)


    clean_iamge,PSNR,PSNR_I1,cost_list=EPLL_half_quadratic_split(noise_image,lamb,patch_size,T,betas,prior,I,log_fuction)

    imsave(f"{save_path}/original_image.jpg",I)
    imsave(f"{save_path}/noise_image.jpg",noise_image)
    imsave(f"{save_path}/denoise_result.jpg",clean_iamge)
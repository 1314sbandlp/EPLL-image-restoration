import numpy as np
from noiseEstimation import estimate_noiseSD_using_Kurts
import utils

def EPLL_half_quadratic_split(noise_image,lamb,patch_size,T,betas=None,prior=None,I=None,log_fuction=None,log=None):
    """EPLL framework
    Args:
        noise_image(numpy.array): the image with noise
        lamb(float/numpy.array):  the parameter lambda from Equation (2) in the paper, if it is matrix, it should have
                                  same size as image
        patch_size(int): the size of patches to extract. patch should be square
        betas(list):
        T(int):
        prior(function):
        I(numpy.array): the original image I, used only for PSNR calculations and comparisons
        Log_fuction(function):
    """
    #print result to log or not
    print_log=log is not None

    #The real image noise standard deviation
    counts = patch_size ** 2
    real_noise_sd=np.sqrt(1/(lamb/counts))

    PSNR=[]
    PSNR_I1=[]
    #If log loss function is given, compute the loss in each iteration
    cost_list=[]
    cal_cost= log_fuction is not None

    #simple guess of beta, in case the auto-estimation doesn't work for some reason
    beta=abs(betas[0]/4)

    #initialize with the noise image
    clean_image=noise_image
    k=1
    sd=np.Inf

    #iteration along all the beta
    for betaa in betas:
        #if betaa<0, estimate beta automatically.
        if betaa<0:
            old_sd=sd
            sd,_=estimate_noiseSD_using_Kurts(clean_image)
            if print_log:
                log.info("sd is %0.4f, beta is %0.4f *(1/real_noise_sd**2) " % (sd,(1/sd**2)/(1/real_noise_sd**2)))
            else:
                print("sd is %0.4f, beta is %0.4f *(1/real_noise_sd**2) " % (sd,(1/sd**2)/(1/real_noise_sd**2)))

            if np.isnan(sd) or sd>old_sd:
                beta=beta*4
                sd=beta**-0.5
            else:
                beta=1/sd**2

        else:
            beta=betaa

        # iteration for T times.
        for i in range(T):
            #get overlapping image patch z from current estimation
            z,_,_=utils.im2col(clean_image,patch_size)
            #compute current cost
            if cal_cost:
                cost_list.append(0.5*np.sum(lamb*np.square(clean_image-noise_image)) - log_fuction(z))
                if print_log:
                    log.info("Cost is %0.4f" % cost_list[-1])
                else:
                    print("Cost is %0.4f" % cost_list[-1])

            #compute MAP estimation on z
            clean_z=prior(z,patch_size,beta**-0.5,noise_image.shape)
            #average the pixels in the cleaned patches in z
            I1=utils.scol2im(clean_z,patch_size,I.shape,"average")
            #close form solution of new estimate image for denoising and inpainting(A is Identity matrix
            clean_image=(noise_image*lamb)/(lamb+beta*counts)+(beta*counts*I1)/(lamb+beta*counts)

            PSNR.append(20*np.log10(1/np.std(clean_image-I)))
            PSNR_I1.append(20*np.log10(1/np.std(I1-I)))
            if print_log:
                log.info("PSNR is %0.4f I1 PSNR: %0.4f" % (PSNR[-1],PSNR_I1[-1] ))
            else:
                print("PSNR is %0.4f I1 PSNR: %0.4f" % (PSNR[-1],PSNR_I1[-1] ))

            k+=1

    clean_image=clean_image.reshape(noise_image.shape)

    # clip values to be between 1 and 0, hardly changes performance
    clean_image[clean_image>1]=1
    clean_image[clean_image<0]=0
    return clean_image,PSNR,PSNR_I1,cost_list



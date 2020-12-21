import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.stats import kurtosis
import math
from utils import rgb2gray
from skimage.io import imread,imsave
from scipy.optimize import fmin
def estimate_noiseSD_using_Kurts(noise_image,patch_size=8):
    """estimate noise level in an image using the kurtosis of marginal filter response distribution
    This code implements the noise estimation method described in:
    "Scale invariance and noise in natural images" by Daniel Zoran and Yair Weiss,
    ICCV 2009
    Args:
        noise_image(numpy.array): image with noise
        patch_size(int): size of DCT patches to use
    :return:
    """

    N=patch_size**2
    #create DCT basis filters
    W,_=DCT_basis(patch_size)

    #remove DC from image(single image)
    meanY=np.mean(noise_image)
    noise_image=noise_image-meanY

    #gather statistics of the image
    noise_vars=np.zeros(N)
    noise_kurts=np.zeros(N)

    for i in range(1,N):
        temp=conv2(noise_image,W[i].reshape(patch_size,patch_size),"valid")
        noise_vars[i]=np.var(temp.reshape(-1))
        noise_kurts[i]=kurtosis(temp.reshape(-1),fisher=False)

    noiseSD, estimated_kurt=fit_shape_SD_using_Kurts(noise_vars,noise_kurts)
    return abs(noiseSD),estimated_kurt

def DCT_basis(patch_size):
    N=patch_size
    W=np.zeros([N**2,N**2])
    k=0
    omega=np.zeros([N**2])
    for p in range(N):
        for q in range(N):
            if p==0:
                ap=1/math.sqrt(N)
            else:
                ap=math.sqrt(2/N)
            if q==0:
                aq=1/math.sqrt(N)
            else:
                aq=math.sqrt(2/N)

            #generate (p,q) filter
            w=np.zeros([N,N])
            for m in range(N):
                for n in range(N):
                    w[m,n]=ap*aq*math.cos(math.pi*(2*m+1)*p/(2*N))*math.cos(math.pi*(2*n+1)*q/(2*N))
            W[k,:]=w.reshape(-1)

            omega[k]=math.sqrt(p*p+q*q)
            k+=1

    index=np.argsort(omega)
    return W[index],omega[index]

def fit_shape_SD_using_Kurts(noise_vars,noise_kurts):
    x=fmin(lambda x: myminfun(x,noise_vars,noise_kurts),[np.min(noise_kurts),0.5])
    noiseSD=x[1]
    kurt=x[0]

    return noiseSD,kurt

def myminfun(guesses,noise_vars,noise_kurts):
    N=noise_vars.shape[0]
    kurt=guesses[0]
    sd=guesses[1]
    F=0
    for i in range(1,N):
        F=F+(kurt_function(kurt,sd,i,noise_vars)-noise_kurts[i])**2
    return F


def kurt_function(clean_kurt,noiseSD,index,vars):
    x=noiseSD**2/(vars[index]-noiseSD**2)
    return (clean_kurt-3)/((1+x)**2)+3


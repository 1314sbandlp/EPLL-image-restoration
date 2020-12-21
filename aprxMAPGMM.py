import numpy as np
import utils
from copy import deepcopy
def aprxMAPGMM(Y,patch_size,noiseSD,imsize,GS,exclude_list=[],sigma_noise=None):
    """approximate MAP estimation with GMM prior
    Args:
        Y(numpy.array): noise patches
        patch_size(int): size of patches(square)
        noiseSD(float):noise standard deviation
        imsize: size of image (not used in this case, but may be used for non local priors)
        GS: learned GMM model
        exclude_list(list):used only for inpainting, misleading name - it's a list
                            of patch indices to use for estimation, the rest are just ignored
        sigma_noise(numpy.array): if the noise is non-white, this is the noise covariance

    """

    if sigma_noise is None:
        sigma_noise=noiseSD**2*np.eye(patch_size**2)

    if len(exclude_list)>0:
        T=Y
        Y=Y[exclude_list,:]

    #remove DC component
    meanY=np.mean(Y,axis=1)[:,np.newaxis]
    Y=Y-meanY

    #calculate assignment probabilities for each mixture component for all patches
    num_component=GS.means_.shape[-1]
    PYZ=np.zeros([num_component,Y.shape[0]])
    GS2=deepcopy(GS)

    for i in range(num_component):
        GS2.covariances_[:,:,i]=GS.covariances_[:,:,i]+sigma_noise
        PYZ[i,:]=np.log(GS.weights_[i])+utils.loggausspdf2(Y,GS2.covariances_[:,:,i])

    #find the most likely component for each patch
    index=np.argmax(PYZ,axis=0)

    #and now perform weiner filtering
    xhat=np.zeros_like(Y)
    for i in range(num_component):
        if sum(index==i)==0:
            continue
        y=np.matmul(GS.covariances_[:,:,i],Y[index==i,:].transpose())+np.matmul(sigma_noise,np.tile(GS.means_[:,i],(sum(index==i),1)).transpose())
        A=GS.covariances_[:,:,i]+sigma_noise
        xhat[index==i,:]=np.linalg.solve(A,y).transpose()

    if len(exclude_list)>0:
        tt=T
        tt[exclude_list,:]=xhat+meanY
        xhat=tt
    else:
        xhat=xhat+meanY

    return xhat





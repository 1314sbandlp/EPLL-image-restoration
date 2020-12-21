import numpy as np
def patch_DC_TGG(Z,patch_size,noiseSD,imsize,W,invW,exlcudeList=[]):
    """independent patch esitmation(ICA prior)
    Args:
        Z(numpy.array): patches set for image
        patch_size(int): the size of patch
        noiseSD(float): standard deviation of noise
        imsize(tuple): size of original image(not used)
        W(numpy.array): ICA parameters
        invW(numpy.array): inverse of W
        exlcudeList(list): exlcude patches, use for inpainting


    """
    #remove DC component
    meanZ=np.mean(Z,axis=1)[:,np.newaxis]
    Z=Z-meanZ

    if len(exlcudeList)==0:
        WZ=np.matmul(W,Z.transpose())
    else:
        WZ=np.matmul(W,Z[exlcudeList,:].transpose())

    t=noiseSD*3
    WZ[abs(WZ)<t]=0
    cleanZ=Z
    if len(exlcudeList)==0:
        cleanZ=np.matmul(invW,WZ)
    else:
        cleanZ[exlcudeList,:]=np.matmul(invW,WZ)
    return cleanZ.transpose()+meanZ



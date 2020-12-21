import numpy as np
from scipy.stats import multivariate_normal
import logging
import tqdm
import os
def im2col(clean_image,patch_size):
    """create overlapping image patches
    Args:
        clean_image(numpy.array): original image
        patch_size(int): size of patch

    """
    H=clean_image.shape[0]
    W=clean_image.shape[1]
    H_len=H-patch_size+1
    W_len=W-patch_size+1
    ih=np.tile(np.arange(patch_size),patch_size)
    iw=np.repeat(np.arange(H_len),W_len)
    i=iw.reshape(-1,1)+ih.reshape(1,-1)

    jh=np.repeat(np.arange(patch_size),patch_size)
    jw=np.tile(np.arange(W_len),H_len)
    j=jw.reshape(-1,1)+jh.reshape(1,-1)

    return clean_image[i,j],i,j


def scol2im(Z,patch_size,img_shape,method,weights=None):
    """ convert image patches back to image
    Args:
        Z(numpy.array): image patches
        patch_size(int): size of patches
        img_shape(tuple): size of original image
        method(str): the method used
        weights(numpy.array): weight in weight average.


    """

    def average():
        #construct average term
        counts=np.zeros(img_shape)
        np.add.at(counts,(i,j),t)
        #sum over patch and do average
        I=np.zeros(img_shape)
        np.add.at(I,(i,j),Z)
        return I/counts

    def sum():

        counts=np.zeros(img_shape)
        np.add.at(counts,(i,j),t)

        I=np.zeros(img_shape)
        np.add.at(I,(i,j),Z)
        return I,counts

    def waverage():

        ws=np.zeros(img_shape)
        np.add.at(ws,(i,j),weights)
        weights_reshape=weights/im2col(ws,patch_size)[0]
        I=np.zeros(img_shape)
        np.add.at(I,(i,j),Z*weights_reshape)

        return I

    def method_select():
        methods={
            "average":average,
            "sum":sum,
            "waverage":waverage
        }

        use_method=methods.get(method)
        if use_method:
            return use_method()
        else:
            return None

    t, i, j = im2col(np.ones(img_shape), patch_size)
    return method_select()


def loggausspdf2(X,sigma):
    """Compute log likelihood for gaussian distribution (zero mean)
    Args:
        X(numpy.array): input vector
        sigma(numpy.array): covariance matrix of gaussian distribution

    Returns:

    """
    d=X.shape[1]
    R=np.linalg.cholesky(sigma).transpose()
    q=np.sum(np.square(np.linalg.solve(R,X.transpose())),axis=0)
    c=d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(R)))
    y = -(c + q) / 2

    return y



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def GMM_log_p(Z,GS):
    """Posterior probability in GMM
    Args:
        Z(numpy.array): patches sample
        GS(class): GMM model

    Returns:

    """
    num_component=GS.means_.shape[-1]
    p=np.zeros(Z.shape[0])
    for i in range(num_component):
        p+=GS.weights_[i]*multivariate_normal.pdf(Z,GS.means_[:,i],GS.covariances_[:,:,i],allow_singular=True)
    return np.sum(np.log(p))



def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
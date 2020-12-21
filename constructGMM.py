
import numpy as np
class GMM_from_learned_model():
    """
    Construct GMM model from data
    """
    def __init__(self,data_path):
        super(GMM_from_learned_model, self).__init__()
        self.weights_=np.loadtxt(f"{data_path}/mixweights.txt")
        self.means_=np.loadtxt(f"{data_path}/means.txt",delimiter=",")
        self.covariances_=np.loadtxt(f"{data_path}/covs.txt",delimiter=",").reshape([64,200,64]).transpose(0,2,1)




from sklearn.decomposition import PCA
import numpy as np


def myPCA(image, pc_num):
    HI_in_column = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    pca = PCA(n_components=pc_num)
    meanval = np.mean(HI_in_column, axis=0)
    meanmat = np.tile(meanval, (image.shape[0] * image.shape[1], 1))
    pca_res = pca.fit_transform(HI_in_column - meanmat)
    return np.reshape(pca_res, (image.shape[0], image.shape[1], pca_res.shape[1]))

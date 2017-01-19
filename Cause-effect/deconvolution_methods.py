'''
Different deconvolution methods implemented
Authors : Olivier Goudet & Diviyan Kalainathan
Date : 15/11/2016
'''
import numpy as np

def deconvolution_methods(deconvolution_method,skel_mat):
    if deconvolution_method == 1:
        """This is a python implementation/translation of network deconvolution

    AUTHORS:
        Algorithm was programmed by Soheil Feizi.
        Paper authors are S. Feizi, D. Marbach,  M. Medard and M. Kellis

    REFERENCES:
       For more details, see the following paper:
        Network Deconvolution as a General Method to Distinguish
        Direct Dependencies over Networks
        By: Soheil Feizi, Daniel Marbach,  Muriel Medard and Manolis Kellis
        Nature Biotechnology"""  # Credits, Ref

        # Gdir = np.dot(skel_mat, np.linalg.inv(np.identity(len(var_names)) + skel_mat))

        """Author code transposed from matlab to python"""
        beta = 0.5
        alpha = 0.1
        # pre - processing the input matrix
        # mapping between 0 and 1
        skel_mat = (skel_mat - np.min(skel_mat)) / (np.max(skel_mat) - np.min(skel_mat))

        # Set diagonal terms to 0
        for idx in range(skel_mat.shape[0]):
            skel_mat[idx, idx] = 0

        # thresholding the input matrix
        y = np.percentile(skel_mat, alpha * 100)
        skel_mat[skel_mat < y] = 0

        D, U = np.linalg.eig(skel_mat)

        lam_n = abs(min(np.min(D), 0))
        lam_p = abs(max(np.max(D), 0))

        m1 = lam_p * (1 - beta) / beta;
        m2 = lam_n * (1 + beta) / beta;
        m = max(m1, m2);

        D = D * np.identity(D.shape[0])

        for i in range(0, D.shape[0]):
            D[i, i] = D[i, i] / (m + D[i, i]);

        mat_new1 = np.dot(np.dot(U, D), np.linalg.inv(U))

        m2 = np.min(mat_new1);
        mat_new2 = (mat_new1 + max(-m2, 0));

        m1 = np.min(mat_new2);
        m2 = np.max(mat_new2);

        Gdir = (mat_new2 - m1) / (m2 - m1);



    elif deconvolution_method == 2:
        """This is a python implementation/translation of network deconvolution

        AUTHORS :
            B. Barzel, A.-L. Barab\'asi

        REFERENCES :
            Network link prediction by global silencing of indirect correlations
            By: Baruch Barzel, Albert-L\'aszl\'o Barab\'asi
            Nature Biotechnology"""  # Credits, Ref
        mat_diag = np.zeros((skel_mat.shape))
        D_temp = np.dot(skel_mat - np.identity(skel_mat.shape[0]), skel_mat)
        for i in range(skel_mat.shape[0]):
            mat_diag[i, i] = D_temp[i, i]
        Gdir = np.dot((skel_mat - np.identity(skel_mat.shape[0]) + mat_diag), np.linalg.inv(skel_mat))

    elif deconvolution_method == 3:
        """Partial correlation coefficient"""
        inv_mat = np.linalg.inv(skel_mat)
        Gdir = np.zeros(inv_mat.shape)
        for i in range(skel_mat.shape[0]):
            for j in range(skel_mat.shape[0]):
                if i != j:
                    Gdir[i, j] = -inv_mat[i, j] / np.sqrt(inv_mat[i, i] * inv_mat[j, j])

    else:
        raise ValueError

    return Gdir


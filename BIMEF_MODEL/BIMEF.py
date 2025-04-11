import numpy as np
import cv2
import time
from imresize import imresize
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy.sparse import spdiags
from scipy.optimize import fminbound


def BIMEF(I, mu=0.5, k=None, a=-0.3293, b=1.1258):
    """
    :param I:   image data (of an RGB image) stored as a 3D numpy array (height x width x color)
    :param mu:  enhancement ratio
    :param k:   exposure ratio (array)
    :param a:   camera response model parameter
    :param b:   camera response model parameter
    :return:    fused: enhanced result
    """

    def maxEntropyEnhance(I, isBad=None):
        Y = rgb2gm(np.real(np.maximum(imresize(I, output_shape=(50, 50)), 0)))

        if not (isBad is None):
            isBad = (255*isBad).astype(np.uint8) 
            isBad = imresize(isBad, output_shape=(50, 50))
            isBad = isBad > 128 
            Y = Y.T[isBad.T]

        if Y.size == 0:
            J = I
            return J

        def find_negative_entropy(Y, k):
            applied_k = applyK(Y, k)
            applied_k[applied_k > 1] = 1
            applied_k[applied_k < 0] = 0
            scaled_applied_k = 255*applied_k + 0.5 
            int_applied_k = scaled_applied_k.astype(np.uint8)
            hist = np.bincount(int_applied_k, minlength=256)
            nonzero_hist = hist[hist != 0]
            normalized_hist = 1.0 * nonzero_hist / applied_k.size
            negative_entropy = np.sum(normalized_hist * np.log2(normalized_hist))
            return negative_entropy

        opt_k = fminbound(func=lambda k: find_negative_entropy(Y, k), x1=1.0, x2=7.0, full_output=False)
        J = applyK(I, opt_k) - 0.01  

        return J

    I = im2double(I)

    lamb = 0.5
    sigma = 5

    t_b = np.amax(I, axis=2)
    t_our = imresize(tsmooth(imresize(t_b, scalar_scale=0.5), lamb, sigma), output_shape=t_b.shape)

    if k is None or k.size == 0:
        isBad = t_our < 0.5  
        J = maxEntropyEnhance(I, isBad)
    else:
        J = applyK(I, k)
        J = np.minimum(J, 1)

    t = np.tile(np.expand_dims(t_our, axis=2), (1, 1, I.shape[2]))  
    W = t ** mu  
    I2 = I * W  
    J2 = J * (1.0-W)  
    fused = I2 + J2  

    fused[fused > 1] = 1  
    fused[fused < 0] = 0  
    fused = (255*fused + 0.5).astype(np.uint8)  
    return fused


def rgb2gm(I):
    if I.shape[2] == 3:
        I = im2double(np.maximum(0, I)) 
        I = (I[:, :, 0] * I[:, :, 1] * I[:, :, 2]) ** (1.0/3.0)  
    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = I**gamma*beta
    return J


def tsmooth(I, lamb=0.01, sigma=3.0, sharpness=0.001):
    I = im2double(I)
    x = I
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamb)
    S = np.squeeze(S)
    return S


def computeTextureWeights(fin, sigma, sharpness):
    v1 = np.diff(fin, axis=0)
    v2 = np.expand_dims(fin[0, :] - fin[-1, :], axis=0)
    dt0_v = np.concatenate((v1, v2), axis=0)
    h1 = np.matrix(np.diff(fin, axis=1)).H
    h2 = np.matrix(np.expand_dims(fin[:, 0], axis=1)).H - np.matrix(np.expand_dims(fin[:, -1], axis=1)).H
    dt0_h = np.matrix(np.concatenate((h1, h2), axis=0)).H

    gauker_h = signal.convolve2d(dt0_h, np.ones((1, sigma)), mode='same')
    gauker_v = signal.convolve2d(dt0_v, np.ones((sigma, 1)), mode='same')
    W_h = np.multiply(np.absolute(gauker_h), np.absolute(dt0_h)) + sharpness
    W_h = np.divide(1, W_h)
    W_v = np.multiply(np.absolute(gauker_v), np.absolute(dt0_v)) + sharpness
    W_v = np.divide(1, W_v)

    return W_h, W_v


def solveLinearEquation(IN, wx, wy, lamb):
    if len(IN.shape) == 2:
        IN = np.expand_dims(IN, axis=2)
    r, c, ch = IN.shape
    k = r * c
    dx = -lamb * np.reshape(wx, (wx.size, 1), order='F')
    dy = -lamb * np.reshape(wy, (wy.size, 1), order='F')
    tempx = np.concatenate((wx[:, -1], wx[:, 0:-1]), axis=1)
    tempy = np.concatenate((np.expand_dims(wy[-1, :], axis=0), wy[0:-1, :]), axis=0)
    dxa = -lamb * np.reshape(tempx, (tempx.size, 1), order='F')
    dya = -lamb * np.reshape(tempy, (tempy.size, 1), order='F')
    tempx = np.concatenate((wx[:, -1], np.zeros((r, c-1))), axis=1)
    tempy = np.concatenate((np.expand_dims(wy[-1, :], axis=0), np.zeros((r-1, c))), axis=0)
    dxd1 = -lamb * np.reshape(tempx, (tempx.size, 1), order='F')
    dyd1 = -lamb * np.reshape(tempy, (tempy.size, 1), order='F')
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamb * np.reshape(wx, (wx.size, 1), order='F')
    dyd2 = -lamb * np.reshape(wy, (wy.size, 1), order='F')

    Ax = spdiags(np.concatenate((dxd1, dxd2), axis=1).T, [-k+r, -r], k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), axis=1).T, [-r+1, -1], k, k)

    D = 1 - (dx + dy + dxa + dya) 

    Axy = Ax + Ay
    A = Axy + Axy.T + spdiags(D.T, 0, k, k)

    fast = True
    if fast:
        OUT = IN
        for ii in range(ch):
            tin = IN[:, :, ii]
            tin = np.reshape(tin, (tin.size, 1), order='F')
            start_cholmod = time.time()
            tout = spsolve(A, tin)
            end_cholmod = time.time()
            time_cholmod = end_cholmod-start_cholmod
            print(time_cholmod) s

            OUT[:, :, ii] = np.reshape(tout, (r, c), order='F')  
    else:
        OUT = IN
        for ii in range(ch):
            tin = IN[:, :, ii]
            tin = np.reshape(tin, (tin.size, 1), order='F')
            tout = np.linalg.lstsq(A.toarray(), tin)
            OUT[:, :, ii] = np.reshape(tout, (r, c), order='F')

    return OUT


def im2double(im):
    if im.dtype == np.float64:
        return im  
    else:
        info = np.iinfo(im.dtype) 
        return im.astype(np.float64) / info.max  

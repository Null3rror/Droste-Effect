import numpy as np
import cv2
import random

repeats = int(input())
r1 = 0.2
r2 = 0.9


def Init(imgAddress):
    img = cv2.imread(imgAddress, cv2.IMREAD_COLOR)
    r = img.shape[0]
    c = img.shape[1]
    x = np.linspace(-1.0, 1.0, num=r, endpoint=True)
    y = np.linspace(-1.0, 1.0, num=c, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X + (1j * Y);
    return img, r, c, Z

def TILE(W, r, c):
    Wx = np.real(W); Wy = np.imag(W)
    wxmax = np.absolute(Wx).max(); wymax = np.absolute(Wy).max()
    Xnew = makeNewXY(Wx, wxmax, c); Ynew = makeNewXY(Wy, wymax, r)
    Xnew = np.tile(Xnew, (repeats, 1)); Ynew = np.tile(Ynew, (repeats, 1))
    for i in range(r *repeats):
        for j in range(c):
            Ynew[i][j] += int(i / c) * r
    Xnew = bringBackXY(Xnew, wxmax, r)
    Ynew = bringBackXY(Ynew, wymax, c)
    return Xnew + (1j * Ynew)

def ROTATION(z, f, tetha):
    return z * f * np.exp(1j * tetha)

def LOG(z):
    zLength = np.absolute(z)
    if min(r1, r2) <= zLength and zLength <= max(r1, r2):
        return(np.log(z/min(r1,r2)))
    return 0

def EXP(z):
    return np.exp(z)

def LogTransform(Z):
    Z1 = Z.copy()
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z1[i][j] = LOG(Z1[i][j])
    return(Z1)

def ExpTransform(Z):
    Z1 = Z.copy()
    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            Z1[i][j] = EXP(Z1[i][j])
    return(Z1)

def RotationTransformPi4(Z):
    Z1 = Z.copy()
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z1[i][j] = ROTATION(Z1[i][j], 1, np.pi/4)
    return(Z1)

def RotationTransform(Z):
    Z1 = Z.copy()
    alpha = np.arctan( ( np.log(max(r2, r1) / min(r1, r2) ) / ( 2 * np.pi ) ) )
    f = np.cos(alpha)
    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            Z1[i][j] = ROTATION(Z1[i][j], f, alpha)
    return(Z1)

def makeNewXY(W, wmax, x):
    return(np.multiply(np.add(np.divide(W, wmax), 1), x/2))
def bringBackXY(W, wmax, x):
    return(np.multiply((np.add(np.divide(W, x/2) ,-1)), wmax))

def recreateImage(X, Y, img, repeat):
    print(X.shape, Y.shape, img.shape)
    X = np.clip(X, 0, img.shape[1]-1)
    Y = np.clip(Y, 0, img.shape[0]-1)
    result = np.zeros(img.shape)
    for j in range(Y.shape[1]):
        for i in range(X.shape[0]):
            result[int(Y[i][j])][int(X[i][j])] = img[i%img.shape[0]][j]
    return result



def CalculateXNewYNewTILED(Z, r, c):
    print(Z.shape)
    W = LogTransform(Z)
    W1 = TILE(W, r, c)
    W1 = RotationTransform(W1)
    W1 = ExpTransform(W1)
    Wx = np.real(W1)
    Wy = np.imag(W1)
    wxmax = np.absolute(Wx).max()
    wymax = np.absolute(Wy).max()
    Xnew = makeNewXY(Wx, wxmax, c)
    Ynew = makeNewXY(Wy, wymax, r)

    return Xnew, Ynew

img, r, c, Z = Init('pb.jpg')
Xnew, Ynew = CalculateXNewYNewTILED(Z, r, c)
newImg = recreateImage(Xnew, Ynew, img, repeats)
cv2.imwrite("droste.jpg", newImg);
input()

import numpy as np
import cv2
import random

repeats = 3
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
    W1 = np.array([ [complex] * c * repeats] * r )
    for i in range(repeats-1):
        if i == 0:
            W1 = np.concatenate([W, W + (i+1) * np.log(r2/r1)], axis = 1)
        else:
            W1 = np.concatenate([W1, W + (i+1) * np.log(r2/r1)], axis = 1)
    return W1;

def ROTATION(z, f, tetha):
    return z * f * np.exp(1j * tetha)

def LOG(z):
    zLength = np.absolute(z)
    if min(r1, r2) <= zLength and zLength <= max(r1, r2):
        return(np.log(z/min(r1,r2)))
    return 0

def EXP(z):
    return np.exp(z)

def LogTransform(Z, r, c):
    Z1 = Z[:r, :c].copy()
    for i in range(r):
        for j in range(c):
            Z1[i][j] = LOG(Z1[i][j])
    return(Z1)

def ExpTransform(Z, r, c):
    Z1 = Z[:r, :c].copy()
    for i in range(r):
        for j in range(c):
            Z1[i][j] = EXP(Z1[i][j])
    return(Z1)

def RotationTransformPi4(Z, r, c):
    Z1 = Z[:r, :c].copy()
    for i in range(r):
        for j in range(c):
            Z1[i][j] = ROTATION(Z1[i][j], 1, np.pi/4)
    return(Z1)

def RotationTransform(Z, r, c):
    Z1 = Z[:r, :c].copy()
    alpha = np.arctan( ( np.log(max(r2, r1) / min(r1, r2) ) / ( 2 * np.pi ) ) )
    f = np.cos(alpha)
    for i in range(r):
        for j in range(c):
            Z1[i][j] = ROTATION(Z1[i][j], f, alpha)
    return(Z1)

def makeNewXY(W, wmax, x):
    return(np.multiply( np.add(np.divide(W, wmax), 1)  , x/2))

def recreateImage(Xnew, Ynew, img, r, c, repeat):
    newImg = np.zeros([r , c*repeat, 3])
    for i in range(r):
        for j in range(c*repeat):
            for k in range(3):
                if int(Xnew[i, j]) == c * repeat:
                    Xnew[i, j] = c * repeat - 1
                if int(Ynew[i, j]) == r:
                    Ynew[i, j] = r  - 1
                newImg[int(Ynew[i, j])][int(Xnew[i, j])][k] = img[i, j % c, k]
    return newImg


def CalculateXNewYNew(Z, r, c):
    W = LogTransform(Z, r, c)
    #W = RotationTransformPi4(Z, r, c)
    Wx = np.real(W)
    Wy = np.imag(W)
    wxmax = np.absolute(Wx).max()
    wymax = np.absolute(Wy).max()
    Xnew = makeNewXY(Wx, wxmax, c)
    Ynew = makeNewXY(Wy, wymax, r)
    return Xnew, Ynew

def CalculateXNewYNewTILED(Z, r, c):
    print(Z.shape)
    W = LogTransform(Z, r, c)
    W1 = TILE(W, r, c)
    print(W1.shape)
    W1 = RotationTransform(W1, r, c * repeats)
    W1 = ExpTransform(W1, r, c * repeats)
    Wx = np.real(W1)
    Wy = np.imag(W1)
    wxmax = np.absolute(Wx).max()
    wymax = np.absolute(Wy).max()
    Xnew = makeNewXY(Wx, wxmax, c * repeats)
    Ynew = makeNewXY(Wy, wymax, r)
    return Xnew, Ynew

img, r, c, Z = Init('PrideCircle.png')
if repeats != 1:
    Xnew, Ynew = CalculateXNewYNewTILED(Z, r, c)
else:
    Xnew, Ynew = CalculateXNewYNew(Z, r, c)
newImg = recreateImage(Xnew, Ynew, img, r, c, repeats)
cv2.imwrite("PrideCircleAfter.png", newImg);
#input()

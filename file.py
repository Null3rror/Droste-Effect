import numpy as np
import cv2
import random

repeats = int(input())

def Init(imgAddress):
    img = cv2.imread(imgAddress, cv2.IMREAD_COLOR)
    r = img.shape[0]
    c = img.shape[1]
    x = np.linspace(-1.0, 1.0, num=r, endpoint=True)
    y = np.linspace(-1.0, 1.0, num=c, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X + (1j * Y);
    return img, r, c, Z

def TILE(z, r1, r2, cOrig, j):
    print(np.imag(z), j)
    z = (1j * np.imag(z)) + np.mod(np.real(z) ,np.log(r2/r1)) * (int(j/cOrig) + 1)
    return z;

def ROTATION(z, f, tetha):
    return z * f * np.exp(1j * tetha)

def LOG(z, r1, r2):
    zLength = np.absolute(z)
    if min(r1, r2) <= zLength and zLength <= max(r1, r2):
        return(np.log(z/min(r1,r2)))
    return 0

def EXP(z):
    return np.exp(z)

def LogTransform(Z, r, c):
    Z1 = Z[:r, :c].copy()
    r1 = 0.2
    r2 = 0.9
    for i in range(r):
        for j in range(c):
            Z1[i][j] = LOG(Z1[i][j], r1, r2)
    return(Z1)

def TILETransform(Z, r, c, cOrig):
    Z1 = Z[:r, :c].copy()
    r1 = 0.2
    r2 = 0.9
    for i in range(r):
        for j in range(c):
            Z1[i][j] = TILE(Z1[i][j], r1, r2, cOrig, j)
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
    r1 = 0.2
    r2 = 0.9
    alpha = np.arctan( ( np.log(max(r2, r1) / min(r1, r2) ) / ( 2 * np.pi ) ) )
    f = np.cos(alpha)
    try:
        for i in range(r):
            for j in range(c):
                Z1[i][j] = ROTATION(Z1[i][j], f, alpha)
        return(Z1)
    except IndexError:
        print(i, j)
def makeNewXY(W, wmax, x):
    return(np.multiply( np.add(np.divide(W, wmax), 1)  , x/2))

def recreateImage(Xnew, Ynew, img, r, c, repeat):
    try:
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
    except IndexError:
        print(int(Ynew[i, j]), int(Xnew[i, j]), i, j, newImg.shape)

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
    Z = np.tile(Z, (1, repeats))
    print(Z.shape)
    W = LogTransform(Z, r, c * repeats)
    W = TILETransform(W, r, c * repeats, c)
    #W = RotationTransform(W, r, c * repeats)
    #W = ExpTransform(W, r, c * repeats)
    Wx = np.real(W)
    Wy = np.imag(W)
    wxmax = np.absolute(Wx).max()
    wymax = np.absolute(Wy).max()
    Xnew = makeNewXY(Wx, wxmax, c * repeats)
    Ynew = makeNewXY(Wy, wymax, r)
    #print(Xnew.max(), Ynew.max())
    return Xnew, Ynew

img, r, c, Z = Init('clock.jpg')
if repeats != 1:
    Xnew, Ynew = CalculateXNewYNewTILED(Z, r, c)
else:
    Xnew, Ynew = CalculateXNewYNew(Z, r, c)
newImg = recreateImage(Xnew, Ynew, img, r, c, repeats)
cv2.imwrite("droste.jpg", newImg);
#input()

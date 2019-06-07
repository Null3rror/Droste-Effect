import numpy as np
import cv2
import random

def ROTATION(z, r, tetha):
    return z * r * np.exp(1j * tetha)

def LOG(z, r1, r2):
    zLength = np.absolute(z)
    if min(r1, r2) <= zLength and zLength <= max(r1, r2):
        return(np.log(z/min(r1,r2)))
    return 0

def WFunc(Z, r, c):
    Z1 = Z[:r, :c].copy()
    r1 = random.random()
    r2 = random.random()

    for i in range(r):
        for j in range(c):
            Z1[i][j] = LOG(Z1[i][j], r1, r2)
            #Z1[i][j] = ROTATION(Z1[i][j], 1, np.pi/4)

    return(Z1)

def makeNew(W, wmax, x):
    return(np.multiply( np.add(np.divide(W, wmax), 1)  , x/2))

def createImageAfterMapping(Xnew, Ynew, img, r, c, repeats):
    newImg = np.zeros([r*repeats, c, 3])
    for i in range(r):
        for j in range(c*repeats):
            for k in range(3):
                if int(Xnew[i, j]) == r:
                    Xnew[i, j] = r - 1
                if int(Ynew[i, j]) == c * repeats:
                    Ynew[i, j] = c * repeats - 1
                newImg[int(Ynew[i, j])][int(Xnew[i, j])][k] = img[i, j % c, k]
    return newImg

img = cv2.imread('clock.jpg', cv2.IMREAD_COLOR)
r = img.shape[0]
c = img.shape[1]
x = np.linspace(-1.0, 1.0, num=r, endpoint=True)
y = np.linspace(-1.0, 1.0, num=c, endpoint=True)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = X + (1j * Y);

W = WFunc(Z, r, c)
Wx = np.real(W)
Wy = np.imag(W)
wxmax = np.absolute(Wx).max()
wymax = np.absolute(Wy).max()
#print(wxmax1, wxmax2)


Xnew = makeNew(Wx, wxmax, c)
Ynew = makeNew(Wy, wymax, r)
newImg = createImageAfterMapping(Xnew, Ynew, img, r, c, 1)
#cv2.imshow('clock', newImg)
cv2.imwrite("seplog.jpg", newImg);
#input()

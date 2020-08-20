import numpy as np
import cv2
import random
import sys
import os.path
# Math explained here: http://www.josleys.com/article_show.php?id=82

class Droste:
    def __init__(self, fPath, repeats, r1, r2, xRange, yRange, resultFolderAddress):
        self.repeats = repeats
        self.r1 = min(r1, r2)
        self.r2 = max(r1, r2)
        self.inputImg = cv2.imread(fPath, cv2.IMREAD_UNCHANGED) #returns a matrix-like object
        self.xRange = xRange
        self.yRange = yRange
        self.resultFolderAddress = resultFolderAddress

    def Transform(self, method):
        # Builds  new plane
        img, r, c, Z = self.CreateComplexPlane()
        if method == 1:
            # 1: Ln -> Tile   -> Rotate -> exponent
            self.Calculate1(Z, r, c)  # takes in the complex plane, outputs the transformed plane
        else:
            # 0: Ln -> Rotate -> Tile   -> exponent
            self.Calculate(Z, r, c)  # takes in the complex plane, outputs the transformed plane


    def CreateComplexPlane(self):
        '''
        :param self: loads the input image
        :return: the image as matrix, nbr of rows, nbr of cols, complex plane
        '''
        img = self.inputImg
        r = img.shape[0] #nbr of rows
        c = img.shape[1] #nbr of cols
        x = np.linspace(-self.xRange, self.xRange, num=r, endpoint=True, dtype=np.float32) #Returns an array of evenly spaced numbers over a specified interval.
        y = np.linspace(-self.yRange, self.yRange, num=c, endpoint=True, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing='ij') #creates rect grid off arrays, here it uses Matrix indexing. Returns X, Y coordinates
        Z = X + (1j * Y); # Transf to the Complex Plane (then we will transform back to Cartesian) 1J is the imaginary part.
        return img, r, c, Z



    def TILE(self, W, r, c, repeatX, repeatY):
        X = W.real
        Y = W.imag

        xOffset = np.log(self.r2 / self.r1)
        repeatYHalf = int((repeatY - 1)/2)
        repeatXHalf = int((repeatX - 1)/2)
        for i in range(-repeatYHalf, repeatYHalf + 1):
            W2 = None
            for j in range(-repeatXHalf, repeatXHalf + 1):
                if 0 == repeatXHalf + j:
                    W2 = X + j * xOffset + (Y + 2 * np.pi * i) * 1j
                else:
                    W2 = np.concatenate([W2, X + j * xOffset + (Y + np.pi*2*i) * 1j], axis=0)
            if 0 == repeatYHalf + i:
                W1 = W2
            else:
                W1 = np.concatenate([W1, W2], axis=1)
        return W1


    def TILE2(self, W, X, Y, repeatX, repeatY):
        X = W.real
        Y = W.imag

        repeatYHalf = int((repeatY - 1)/2)
        repeatXHalf = int((repeatX - 1)/2)

        alpha = np.arctan2(np.log(max(self.r2,self.r1) / min(self.r1, self.r2)), 2 * np.pi) # we need alpha to calculate the offsets
        cosine = np.cos(alpha)
        sine = np.sin(alpha)
        yOffset = 2 * np.pi * sine * sine
        xOffset = 2 * np.pi * sine * cosine

        for i in range(-repeatYHalf, repeatYHalf + 1):
            W2 = None
            for j in range(-repeatXHalf, repeatXHalf + 1):
                if 0 == repeatXHalf + j:
                    W2 = X + j * xOffset + ((Y + (2 * np.pi * i + yOffset * j)) * 1j)
                else:
                    W2 = np.concatenate([W2, X + j * xOffset + ((Y + (2 * np.pi * i + yOffset * j)) * 1j)], axis=0)
            if 0 == repeatYHalf + i:
                W1 = W2
            else:
                W1 = np.concatenate([W1, W2], axis=1)

        return W1



    def LogTransform(self, Z, r, c):
        '''
        :param Z: Complex plane
        :param r: nbr of rows
        :param c: nbr of cols
        :return: np.log(z / min(r1, r2))
        '''
        Z1 = np.zeros([r, c], dtype=complex)

        ZAbs = np.absolute(Z)
        elemetsToApplyTransformation = (self.r1 <= ZAbs) & (ZAbs <= self.r2)

        Z1[elemetsToApplyTransformation] = np.log(Z[elemetsToApplyTransformation])

        return Z1


    def ExponentTransform(self, Z, r, c, repeatX, repeatY):
        Z1 = Z[:r * repeatX, :c * repeatY].copy()
        Z1 = np.exp(Z)
        return Z1


    def RotationTransform(self, Z, r, c, repeatX, repeatY):
        Z1 = Z[:r * repeatX, :c * repeatY].copy()
        alpha = np.arctan2(np.log(self.r2 / self.r1), 2 * np.pi)
        f = np.cos(alpha)

        Z1 = Z * f * np.exp(1j * alpha)
        return Z1


    def Calculate(self, Z, r, c):
        print(Z.shape)
        W1 = self.LogTransform(Z, r, c)
        self.ReCreateImageAndSave(W1, "log-{}.{}-{}.{}.jpg".format(0, self.r1, self.r2, self.repeats), r, c, 1, 1)

        W1 = self.RotationTransform(W1, r, c, 1, 1)
        self.ReCreateImageAndSave(W1, "rot-{}.{}-{}.{}.jpg".format(0, self.r1, self.r2, self.repeats), r, c, 1, 1)

        repeatX = self.repeats
        repeatY = 1
        W1 = self.TILE2(W1, r, c, repeatX, repeatY)
        print(W1.shape)
        self.ReCreateImageAndSave(W1, "tile-{}.{}-{}.{}.jpg".format(0, self.r1, self.r2, self.repeats), r, c, repeatX, repeatY)

        W1 = self.ExponentTransform(W1, r, c, repeatX, repeatY)
        self.ReCreateImageAndSave(W1, "final-{}.{}-{}.{}.jpg".format(0, self.r1, self.r2, self.repeats), r, c, repeatX, repeatY)

    def Calculate1(self, Z, r, c):
        print(Z.shape)
        # apply Ln(z/r1)
        W1 = self.LogTransform(Z, r, c)
        self.ReCreateImageAndSave(W1, "log-{}.{}-{}.{}.jpg".format(1, self.r1, self.r2, self.repeats), r, c, 1, 1)

        # choose how many tiles we want to add
        repeatX = self.repeats
        repeatY = 1
        # # apply Tile
        W1 = self.TILE(W1, r, c, repeatX, repeatY)
        print(W1.shape)
        self.ReCreateImageAndSave(W1, "tile-{}.{}-{}.{}.jpg".format(1, self.r1, self.r2, self.repeats), r, c, repeatX, repeatY)
        #
        # # apply Rotation
        W1 = self.RotationTransform(W1, r, c, repeatX, repeatY)
        self.ReCreateImageAndSave(W1, "rot-{}.{}-{}.{}.jpg".format(1, self.r1, self.r2, self.repeats), r, c, repeatX, repeatY)

        # apply exponent
        W1 = self.ExponentTransform(W1, r, c, repeatX, repeatY)
        self.ReCreateImageAndSave(W1, "final-{}.{}-{}.{}.jpg".format(1, self.r1, self.r2, self.repeats), r, c, repeatX, repeatY)


    def makeNewXY(self, W, wmax, x):
        return np.multiply(np.add(np.divide(W, wmax), 1), x / 2)

    def GetNewXY(self, W, r, c, repeatX, repeatY):
        Wx = np.real(W)
        Wy = np.imag(W)
        wxmax = np.absolute(Wx).max()
        wymax = np.absolute(Wy).max()
        XNew = self.makeNewXY(Wx, wxmax, c * repeatY)
        YNew = self.makeNewXY(Wy, wymax, r * repeatX)
        return XNew, YNew

    def ReCreateImage(self, Xnew, Ynew, img, r, c, repeatX, repeatY):
        '''
        :param Xnew:
        :param Ynew:
        :param img:
        :param r:
        :param c:
        :param repeat:
        :return: A matrix representing the pic in the new coordinates
        '''
        newImg = np.zeros([r * repeatX, c * repeatY, 3])
        for i in range(r * repeatX):
            for j in range(c * repeatY):
                if int(Xnew[i, j]) == c * repeatY:
                    Xnew[i, j] = c * repeatY - 1
                if int(Ynew[i, j]) == r * repeatX:
                    Ynew[i, j] = r * repeatX - 1
                newImg[int(Ynew[i, j])][int(Xnew[i, j])] = img[i % r, j % c] # extract RGB from inputImage

        return newImg



    def ReCreateImageAndSave(self, W, imgTitle, r, c, repeatX, repeatY):
        xNew, yNew = self.GetNewXY(W, r, c, repeatX, repeatY)
        img = self.ReCreateImage(xNew, yNew, self.inputImg, r, c, repeatX, repeatY)
        cv2.imwrite(self.resultFolderAddress + '/' + imgTitle, img)

def GetCommandLineArgs():
    '''
    Gets command line argumets, does some error checking.
    :return: command line arguments
             fPath must be valid
             method must be 0 or 1
             repeats must be positive and odd
    '''
    if len(sys.argv) == 4:
        fPath = sys.argv[1] # Get input File name
        method = int(sys.argv[2])
        repeats = int(sys.argv[3])
        if not os.path.isfile(fPath):
            sys.exit("Input picture does not exist")
        if not (method == 0 or method == 1):
            sys.exit("Method must be 0 or 1")
        if repeats <= 0 or repeats & 1 == 0:
            sys.exit("Repeats must be positive and odd")
        return fPath, method, repeats
    else:
        sys.exit("Need only two command line argument and that's the address of input picture and method number\nMethod number is either 0 or 1\nRepeats must be positive and odd")

def CreateOutputFolder(inputImagePath, repeats, method, r1, r2):
    '''
    :param inputImagePath: address of our input image
    :param repeats:        number of repeats
    :return: address of the result folder
    '''
    inputImageName = inputImagePath.split('/')[-1] # extract input image name
    outputFolder = "OutputImages/"        # pictures will be stored

    resultFolderAddress = outputFolder + inputImageName + "-{}.{}-{}.{}.jpg".format(method, r1, r2, repeats) # the folder that our code is stored in
    if not os.path.exists(resultFolderAddress): # create the result folder
        os.makedirs(resultFolderAddress)
    return resultFolderAddress

def main():
    inputImagePath, method, repeats = GetCommandLineArgs()
    # We have two methods of calculating the result
    # 0: Ln -> Rotate -> Tile   -> exponent
    # 1: Ln -> Tile   -> Rotate -> exponent

    r1      = 0.2      # inner circle radius
    r2      = 0.99    # outer circle radius

    resultFolderAddress = CreateOutputFolder(inputImagePath, repeats, method, r1, r2)

    d = Droste(inputImagePath, repeats, r1, r2, -1, 1, resultFolderAddress)

    d.Transform(method)


if __name__ == "__main__":
    main()

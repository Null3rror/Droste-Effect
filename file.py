import numpy as np
import cv2
import random
# Math explained here: http://www.josleys.com/article_show.php?id=82

class droste:
    def __init__(self, fPath, repeats=3,r1=0.2,r2=0.9):
        self.fPath = fPath
        self.repeats = 3 # how many turns - Make it >1
        self.r1 = r1
        self.r2 = r2

    def newImg(self):
        # Builds  new plane (calls all functions!)

        img, r, c, Z = self.Init()
        if self.repeats != 1:
            Xnew, Ynew = self.CalculateXNewYNewTILED(Z, r,c)  # takes in the complex plane, outputs the transformed plane
        else:
            Xnew, Ynew = self.CalculateXNewYNew(Z, r, c)  # Only the Log transform!

        # Make transform
        newImg = self.recreateImage(Xnew, Ynew, img, r, c, self.repeats)  # takes in new plane and projects the image on it
        return newImg

    def Init(self):
        '''
        :param self: loads the input image
        :return: the image as matrix, nbr of rows, nbr of cols, complex plane
        '''
        img = cv2.imread(self.fPath, cv2.IMREAD_UNCHANGED) #returns a matrix-like object
        r = img.shape[0] #nbr of rows
        c = img.shape[1] #nbr of cols
        x = np.linspace(-1.0, 1.0, num=r, endpoint=True) #Returns an array of evenly spaced numbers over a specified interval.
        y = np.linspace(-1.0, 1.0, num=c, endpoint=True)
        X, Y = np.meshgrid(x, y, indexing='ij') #creates rect grid off arrays, here it uses Matrix indexing. Returns X, Y coordinates
        Z = X + (1j * Y); # Transf to the Complex Plane (then we will transform back to Cartesian) 1J is the imaginary part.
        return img, r, c, Z


    def TILE(self,W, r, c):
        W1 = np.array([[complex] * c * self.repeats] * r)
        for i in range(self.repeats - 1):
            if i == 0:
                W1 = np.concatenate([W, W + (i + 1) * np.log(self.r2 / self.r1)], axis=1)
            else:
                W1 = np.concatenate([W1, W + (i + 1) * np.log(self.r2 / self.r1)], axis=1)
        return W1;


    def LOG(self,z):
        '''
        :param z: Complex plane
        :return: log(z/min(r1,r2))
        '''
        zLength = np.absolute(z)
        if min(self.r1, self.r2) <= zLength and zLength <= max(self.r1, self.r2):
            return (np.log(z / min(self.r1, self.r2)))
        return 0

    def LogTransform(self,Z, r, c):
        '''
        :param Z: Complex plane
        :param r: nbr of rows
        :param c: nbr of cols
        :return: np.log(z / min(r1, r2))
        '''
        Z1 = Z[:r, :c].copy()
        for i in range(r):
            for j in range(c):
                Z1[i][j] = self.LOG(Z1[i][j])
        return (Z1)


    def ExpTransform(self,Z, r, c):
        Z1 = Z[:r, :c].copy()
        for i in range(r):
            for j in range(c):
                Z1[i][j] = np.exp(Z1[i][j])
        return (Z1)


    def ROTATION(self,z, f, tetha):
        return z * f * np.exp(1j * tetha)

    def RotationTransformPi4(self,Z, r, c):
        Z1 = Z[:r, :c].copy()
        for i in range(r):
            for j in range(c):
                Z1[i][j] = self.ROTATION(Z1[i][j], 1, np.pi / 4)
        return (Z1)


    # ROTATE
    def RotationTransform(self,Z, r, c):
        Z1 = Z[:r, :c].copy()
        alpha = np.arctan((np.log(max(self.r2,self.r1) / min(self.r1, self.r2)) / (2 * np.pi)))
        f = np.cos(alpha)
        for i in range(r):
            for j in range(c):
                Z1[i][j] = self.ROTATION(Z1[i][j], f, alpha)
        return (Z1)


    def makeNewXY(self,W, wmax, x):
        return (np.multiply(np.add(np.divide(W, wmax), 1), x / 2))


    def recreateImage(self, Xnew, Ynew, img, r, c, repeat):
        '''
        :param Xnew:
        :param Ynew:
        :param img:
        :param r:
        :param c:
        :param repeat:
        :return: A matrix representing the pic in the new coordinates
        '''
        newImg = np.zeros([r, c * repeat, 3]) #
        for i in range(r):
            for j in range(c * repeat):
                for k in range(3):
                    if int(Xnew[i, j]) == c * repeat:
                        Xnew[i, j] = c * repeat - 1
                    if int(Ynew[i, j]) == r:
                        Ynew[i, j] = r - 1
                    newImg[int(Ynew[i, j])][int(Xnew[i, j])][k] = img[i, j % c, k]
        return newImg


    def CalculateXNewYNew(self,Z, r, c):
        W = self.LogTransform(Z, r, c)
        # W = RotationTransformPi4(Z, r, c)
        Wx = np.real(W)
        Wy = np.imag(W)
        wxmax = np.absolute(Wx).max()
        wymax = np.absolute(Wy).max()
        Xnew = self.makeNewXY(Wx, wxmax, c)
        Ynew = self.makeNewXY(Wy, wymax, r)
        return Xnew, Ynew


    def CalculateXNewYNewTILED(self, Z, r, c):
        print(Z.shape)
        W = self.LogTransform(Z, r, c)
        W1 = self.TILE(W, r, c)
        print(W1.shape)
        W1 = self.RotationTransform(W1, r, c * self.repeats)
        W1 = self.ExpTransform(W1, r, c * self.repeats)
        Wx = np.real(W1)
        Wy = np.imag(W1)
        wxmax = np.absolute(Wx).max()
        wymax = np.absolute(Wy).max()
        Xnew = self.makeNewXY(Wx, wxmax, c * self.repeats)
        Ynew = self.makeNewXY(Wy, wymax, r)
        return Xnew, Ynew


def main():
    fPath = 'INPUT.png'
    repeats = 3
    r1      = 0.2
    r2      = 0.9

    d= droste(fPath, repeats,r1,r2)
    newImg = d.newImg()

    # Saves output  - the transformed picture
    cv2.imwrite("OUTPUT.png", newImg)
    # Display resulting image
    window_name = 'new image - (some colors might be off)'
    cv2.imshow(window_name,newImg)
    # waits for user to press any key
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import numpy as np
import cv2
from PIL import Image
import logging


deformation_scale = 256 #TODO: for Dmytro, this can be changed
droste_scale = 256


def escher_deformation(x,y):
    '''
    Takes Cartesian coordinates, converts into polar, applies transform and returns cartesian outputs
    '''

    #FROM CARTESIAN TO COMPLEX
    Z = x+ y * 1j 

    #APPLY LOG TO THE COMPLEX PLANE
    lnz = np.log(Z) 

    #APPLY ESCHER TRANSFORM
    # need the value of gamma from #3 in https://github.com/lcipolina/escher/blob/Lucia/Math%20Behind%20Escher.pdf
    sn = 1 - (1j * np.log(deformation_scale)) / (2 * np.pi)

    #Rotation
    lnz_sn = lnz * sn
    
    #Exponentiation
    ez = np.exp(lnz_sn)

    #DISPLAY - choose one to see output #TODO: JIM if you change this, you get the 3 steps
    #transform = lnz    #displays the log in the complex plane
    #transform = lnz_sn #displays the translated
    transform = ez       #displays the final exponential

    #Back to Cartesians for display
    Xnew = np.real(transform).astype(np.float32)
    Ynew = np.imag(transform).astype(np.float32)

    return Xnew, Ynew


def escher_reverse(x,y):
    '''
    Takes Cartesian coordinates, converts into polar, applies transform and returns cartesian outputs
    '''

    #FROM CARTESIAN TO COMPLEX
    Z = x+ y * 1j

    #APPLY LOG TO THE COMPLEX PLANE
    lnz = np.log(Z)

    #APPLY ESCHER TRANSFORM
    sn = (2 * np.pi * 1j) / (2 * np.pi * 1j + np.log(deformation_scale))
    # note that this sn is equal to the inverse of the sn used in the escher_deformation function.
    # as explained in the last page of Lucia's pdf, the inverse needs 1 / gamma.

    #Rotation
    lnz_sn = lnz * sn

    #Exponentiation
    ez = np.exp(lnz_sn)


    #DISPLAY - choose one to see output
    #transform = lnz    #displays the log in the complex plane
    #transform = lnz_sn #displays the translated
    transform = ez       #displays the final exponential

    #Back to Cartesians for display
    Xnew = np.real(transform).astype(np.float32)
    Ynew = np.imag(transform).astype(np.float32)

    return Xnew, Ynew


def droste_transformation(x, y, c=1):
    for _ in range(2):
        # if(any(greaterThan(abs(uv),vec2(1.)))):
        indx = (np.abs(x) >= c) | (np.abs(y) >= c)
        # uv *= (1./drostescale);
        x[indx] *= 1. / droste_scale
        y[indx] *= 1. / droste_scale

        # if(all(lessThan(abs(uv),vec2(1./drostescale)))):
        indx = (np.abs(x) < c / droste_scale) & (np.abs(y) < c / droste_scale)
        # uv *= drostescale;
        x[indx] *= droste_scale
        y[indx] *= droste_scale
    return x, y


inFile  = 'images/circle.png'
outFile = '/tmp/deform.png'
outFile2 = '/tmp/reverse.png'

src = np.array(cv2.imread(inFile))

src_row, src_col = src.shape[0], src.shape[1]
out_row = src_row * 20  # multiply by 20 to get higher resolution results
out_col = src_row * 20

range_ = 1 # increasing this value from 1 puts more of the circle on the deformed output but also rotates it
vec1 = np.linspace(-range_, range_, num=out_row, endpoint=True)
vec2 = np.linspace(-range_, range_, num=out_col, endpoint=True)
x, y = np.meshgrid(vec2, vec1)

# to do the forward deformation, use the reverse function
x1, y1 = escher_reverse(x, y)
c = 0.5
x1, y1 = droste_transformation(x1, y1, c)
x1 = (x1 / c + 1) * src_col / 2
y1 = (y1 / c + 1) * src_row / 2
out = cv2.remap(src, x1, y1, interpolation=cv2.INTER_CUBIC, borderValue=100)
cv2.imwrite(outFile, out)

range_ = 1
vec1 = np.linspace(-range_, range_, num=out_row, endpoint=True)
vec2 = np.linspace(-range_, range_, num=out_col, endpoint=True)
x, y = np.meshgrid(vec2, vec1)

# to do the reverse deformation, use the forward function
x2, y2 = escher_deformation(x, y)
c = 1
x2, y2 = droste_transformation(x2, y2, c)
x2 = (x2 / c + 1) * out_col / 2
y2 = (y2 / c + 1) * out_row / 2
out2 = cv2.remap(out, x2, y2, interpolation=cv2.INTER_CUBIC, borderValue=200)
cv2.imwrite(outFile2, out2)

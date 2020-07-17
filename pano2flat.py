import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Reference:
https://www.cnblogs.com/riddick/p/10258216.html
https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map
'''

class Pano:
    __epsion = 10e-7
    def __init__(self, diretory, fov = 90, viewpoint = (0., 0.), f = 1.):
        assert(f > 0.1 and f < 10.)
        assert(fov > 0 and fov < 180)

        self.diretory = diretory
        self.fov = fov
        self.theta = viewpoint[0] * np.pi / 180.
        self.sigma = viewpoint[1] * np.pi / 180.
        self.f = f

        # input image checking
        #with cv2.imread(self.diretory) as img:
        img = cv2.imread(self.diretory)
        self.img = img.copy()
        self.height, self.width, self.channel = self.img.shape 
        assert(abs(self.width - 2 * self.height - 1) <= 2)
        assert(self.channel == 3)
        
        
        #output image
        self.sizeOriginalOutputImage()

    def sizeOriginalOutputImage(self):
        self.r = self.height / np.pi 
        self.size = self.size = 2. * self.r
        self.size = 2.*self.r * np.tan(self.fov * np.pi / 360.)
        self.r += self.__epsion
        self.size = int(self.size)
    
    def show(self):
        print("input image shape", self.img.shape) 
        print("output image size {size}".format(size = self.size))
        print(self.img)
        cv2.imshow('look', self.img)
        cv2.waitKey(0)

    
    def bilinearInterp(self, x, y, a):
        assert(x >= 0 and x < a.shape[1] and y >= 0 and y < a.shape[0])
        x0 = int(x) % a.shape[1]
        y0 = int(y) % a.shape[0]
        x1 = (x0 + 1) % a.shape[1]       # coords of pixel to bottom right
        y1 = (y0 + 1) % a.shape[0]
        nx = x - x0      # fraction of way across pixel
        ny = y - y0
        
        # Pixel values of four corners
        t_l = a[y0, x0]
        t_r = a[y0, x1]
        b_l = a[y1, x0]
        b_r = a[y1, x1]
        
        # interpolate
        res = (t_l*(1-ny)*(1-nx) + t_r*(1-ny)* nx + b_l*ny*(1-nx) + b_r*ny*nx).astype(np.uint8)
        return res

    def pano2flat(self):
        o_img = np.empty([self.size, self.size, self.channel], dtype = np.uint8) 
        ''' 
        x,y,z coordinate 
        x = 0.5* size
        y = 0.5 * size - j
        z = 0.5 * size - i
        '''
        x = 1.
        scale = np.tan(self.fov * np.pi / 360.)
        for i in range(self.size):
            z = scale - i / self.r
            for j in range(self.size):
                y = scale - j / self.r
                # spherial coordinate
                theta = np.mod(np.arctan2(y, x) - self.theta, 2*np.pi) 
                sigma = np.mod(np.arccos(z/np.sqrt(x*x+y*y+z*z)) - self.sigma, 2 * np.pi) 
                # local coordinate at input image
                u = theta * self.width / (2. * np.pi)
                v = sigma * self.height / np.pi
                if (i > 500 and i < 505 and j > 300 and j < 320):
                    print((theta, sigma))
                # Use bilinear interpolation between the four surrounding pixels
                o_img[i,j] = self.bilinearInterp(u, v, self.img)

        return o_img


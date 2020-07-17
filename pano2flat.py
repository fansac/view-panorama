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
    def __init__(self, diretory, f = 1., fov = 0):
        assert(f > 0.1 and f < 10.)
        assert(fov >= 0 and fov < 180)
        self.diretory = diretory
        self.f = f
        self.fov = fov

        # input image checking
        img = cv2.imread(self.diretory)
        self.img = img.copy()
        self.height, self.width, _ = self.img.shape 
        assert(abs(self.width - 2 * self.height - 1) <= 2) 
        
        #output image
        self.sizeOriginalImage()

    def sizeOriginalImage(self):
        self.size = int(self.width / np.pi)
        if self.fov != 0:
            self.size *= np.tan(0.5 * self.fov)
            
    
    def show(self):
        print("input image shape", self.img.shape) 
        print("output image size {size}".format(size = self.size))
        print(self.img[0:10, 0:10])
        cv2.imshow('look', self.img)
        cv2.waitKey(0)

    def pano2flat(self):
        output_image = np.empty([self.size, self.size, 3], dtype = np.uint8) 
        ''' 
        x,y,z coordinate 
        x = 0.5* size
        y = 0.5 * size - j
        z = 0.5 * size - i

        '''
        x = 1.
        r = 0.5 * self.size + self.__epsion
        if r == 0:
            print("error zero")
        for i in range(self.size):
            z = 1 - i / r
            for j in range(self.size):
                try:
                    y = 1 - j / r
                except RuntimeWarning:
                    print("zero error:", i, j)
                # spherial coordinate
                theta = np.mod(np.arctan2(y, x), 2*np.pi)
                sigma = np.arccos(z/np.sqrt(x*x+y*y+z*z)) 
                # local coordinate at input image
                u = theta * self.width / (2. * np.pi)
                v = sigma * self.height / np.pi
                    
                # Use bilinear interpolation between the four surrounding pixels
                ui = int(np.floor(u))  # coord of pixel to bottom left
                vi = int(np.floor(v))
                
                u2 = ui+1       # coords of pixel to top right
                v2 = vi+1
                nu = u-ui      # fraction of way across pixel
                nv = v-vi
                
                # Pixel values of four corners
                A = self.img[vi % self.height,ui % self.width]
                B = self.img[v2 % self.height,ui % self.width]
                C = self.img[vi % self.height,u2 % self.width]
                D = self.img[v2 % self.height,u2 % self.width]
              
                # interpolate
                output_image[i,j]  = (A*(1-nv)*(1-nu) + B*(nv)*(1-nu) + C*(1-nv)*nu+D*nv*nu).astype(np.uint8)

        return output_image
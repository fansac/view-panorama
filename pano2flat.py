import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Reference:
https://www.cnblogs.com/riddick/p/10258216.html
https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map

no enough testing
'''

class Pano:
    __epsion = 10e-7
    def __init__(self, diretory, fov = 90, viewpoint = (0., 0.), f = 1.):
        assert(f > 0.1 and f < 10.)
        assert(fov > 0 and fov < 180)

        self.diretory = diretory
        self.fov = fov
        self.f = f
        self.rotation = self.yzRotate(viewpoint)
        # input image checking
        #with cv2.imread(self.diretory) as img:
        img = cv2.imread(self.diretory)
        self.img = img.copy()
        self.height, self.width, self.channel = self.img.shape 
        assert(abs(self.width - 2 * self.height) <= self.__epsion)
        assert(self.channel == 3)
                
        #output image
        self.sizeOriginalOutputImage()

    def sizeOriginalOutputImage(self):
        self.r = self.height / np.pi 
        self.size = 2.*self.r * np.tan(self.fov * np.pi / 360.)
        self.size = int(self.size)

    def yzRotate(self, viewpoint):
        vp = np.mod(np.array(viewpoint) * np.pi / 180, 2*np.pi)
        rotation = np.zeros((2,3,3))
        rotation[0] = np.array([[np.cos(vp[0]), 0, np.sin(vp[0])],
                                [0,1,0],
                                [-np.sin(vp[0]), 0, np.cos(vp[0])]])
        rotation[1] = np.array([[np.cos(vp[1]), -np.sin(vp[1]), 0],
                                [np.sin(vp[1]), np.cos(vp[1]), 0],
                                [0,0,1]])
        return rotation
    
    def show(self):
        print("input image shape", self.img.shape) 
        print("output image size {size}".format(size = self.size))
        print(self.img)
        cv2.imshow('look', self.img)
        cv2.waitKey(0)

    
    def bilinearInterp(self, x, y, a):
        x0 = (x.astype(np.int)) % a.shape[1]
        y0 = (y.astype(np.int)) % a.shape[0]
        x1 = (x0 + 1) % a.shape[1]       # coords of pixel to bottom right
        y1 = (y0 + 1) % a.shape[0]
        nx = x - x0      # fraction of way across pixel
        ny = y - y0
        
        # Pixel values of four corners
        t_l = a[y0, x0]
        t_r = a[y0, x1]
        b_l = a[y1, x0]
        b_r = a[y1, x1]
        
        # interpolate (for vetorization, need transpose)
        res = (t_l.T*(1-ny).T*(1-nx).T + t_r.T*(1-ny).T* nx.T + b_l.T*ny.T*(1-nx).T + b_r.T*ny.T*nx.T).astype(np.uint8)
        return res.T

    def pano2flat(self):
        o_img = np.empty([self.size, self.size, self.channel], dtype = np.uint8) 
        ''' 
        x,y,z coordinate 
        x = 0.5* size
        y = 0.5 * size - j
        z = 0.5 * size - i
        '''
        x0 = 1.
        scale = np.tan(self.fov * np.pi / 360.)
        for i in range(self.size):
            z0 = scale - i / self.r
            for j in range(self.size):
                y0 = scale - j / self.r
                x, y, z = np.dot(self.rotation[1], np.dot(self.rotation[0], np.array([x0, y0, z0])))
                # spherial coordinate
                theta = np.mod(np.arctan2(y, x), 2*np.pi) 
                sigma = np.mod(np.arccos(z/np.sqrt(x*x+y*y+z*z)), 2 * np.pi) 
       
                # local coordinate at input image
                u = theta * self.width / (2. * np.pi)
                v = sigma * self.height / np.pi

                # Use bilinear interpolation between the four surrounding pixels
                o_img[i,j] = self.bilinearInterp(u, v, self.img)
  
                if i == 10 and j == 10:
                    print("1", (x0,y0,z0), (x, y, z), theta, sigma)
        return o_img

    def pano2flatVectorization(self):
        o_img = np.empty([self.size, self.size, self.channel], dtype = np.uint8) 
        scale = np.tan(self.fov * np.pi / 360.)

        x0 = np.ones(o_img.shape[0:2])
        y0 = scale - np.repeat(np.array([range(self.size)]), repeats = [self.size], axis = 0) / self.r
        z0 = y0.T
        a0 = np.dstack((x0, y0, z0))
        # apply ratation
        a = np.dot(np.dot(a0, self.rotation[0]), self.rotation[1])
        # spherial coordinate
        theta = np.mod(np.arctan2(a[:,:,1], a[:,:,0]), 2*np.pi)
        sigma = np.mod(np.arccos(a[:,:,2]/np.sqrt(np.sum(a*a, axis = 2))), 2 * np.pi)
        # local coordinate at input image
        u = theta * self.width / (2. * np.pi)
        v = sigma * self.height / np.pi
        # Use bilinear interpolation between the four surrounding pixels
        return self.bilinearInterp(u, v, self.img)


   
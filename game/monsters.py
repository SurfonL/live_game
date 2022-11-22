import cv2
import numpy as np

class Monster:
    def __init__(self,monster,mask) -> None:
        self.monster = monster
        self.mask = mask
    
    def resize(self, h,w):
        
        
        try:
            mh,mw,mc = self.monster.shape
            ratio = w/mw
            h = int(mh*ratio)
            mon = cv2.resize(self.monster,(w,h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(self.mask,(w,h), interpolation=cv2.INTER_NEAREST)
            self.monster_resized = mon*mask
            self.mask_resized = mask
        except Exception as e:
            print('resize error', e)
            
    def draw(self, image, posy, posx, h, w):
        self.resize(h,w)
        
        
        try:
            h,w,c = image.shape
            hm, wm, cm = self.monster_resized.shape
            yp0 = int(posy) - int(hm/2)
            yp1 = yp0+hm
            xp0 = int(posx) - int(wm/2)
            xp1 = xp0+wm
            
            
            mask = np.zeros_like(image)
            mon_resized = np.zeros_like(image)
            mask[yp0:yp1,xp0:xp1,:] = self.mask_resized
            mon_resized[yp0:yp1,xp0:xp1,:] = self.monster_resized
            rev = cv2.bitwise_not(mask)
            
            t1 = cv2.bitwise_and(image,rev)
            t2 = mon_resized
            image = cv2.add(t1,t2)
            return image
        except Exception as e:
            print('draw error', e)
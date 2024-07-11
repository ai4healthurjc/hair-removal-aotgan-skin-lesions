import numpy as np
import cv2

def DullRazor(I, M):   

    final_image = cv2.inpaint(I, M, 1,cv2.INPAINT_TELEA)
    
    return final_image

def HairRemovMed_hyadamhuang (I, M, s):
    m, n = M.shape[:2]
    I1 = np.copy(I)
    for i in range(m):
        for j in range(n):
            if M[i, j] > 0:
                x0 = max(i - s, 0)
                x1 = min(i + s, m-1)
                y0 = max(j - s, 0)
                y1 = min(j + s, n-1)
                region = I[x0:x1+1, y0:y1+1]
                median_color = np.median(region, axis=(0,1))
                I1[i, j] = median_color.astype(np.uint8)
    return I1

from collections import deque
from PIL import Image
import numpy as np

def connected_component(sx, sy, img):
    img = np.copy(img)
    rows, cols = img.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque()
    queue.append((sx, sy))
    img[sx, sy] = False
    
    cc = []
    while queue:
        x, y = queue.popleft()
        cc.append((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and img[nx, ny]:
                img[nx, ny] = False
                queue.append((nx, ny))
    
    return cc, img

def connected_components (img) : 
    ccs = [] 
    while np.any(img) : 
        X, Y = np.where(img)
        sx, sy = X[0], Y[0] 
        cc, img = connected_component(sx, sy, img)
        ccs.append(cc)
    return ccs

if __name__ == "__main__" : 
    LAYOUT = np.array(Image.open('munch.png').convert('RGB'))
    HOUSE = np.array([255, 255, 255]) 
    img = np.all(LAYOUT == HOUSE, axis=2)
    print(connected_components(img))

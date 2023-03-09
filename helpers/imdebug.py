from matplotlib import pyplot as plt
import time 
import os
import numpy as np 
import uuid

def imdebug(images):
    
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    if not isinstance(images,list):
        images=[images,]
        
    fig, ax = plt.subplots(1,3,figsize=(10,3))

    for i, im in enumerate(images):

        ax[0].imshow(im[:,:,int(np.floor(im.shape[2]/2))],alpha=0.5,cmap=cmaps[i])
        ax[1].imshow(np.rot90(im[:,int(np.floor(im.shape[1]/2)),:]),alpha=0.5,cmap=cmaps[i])
        ax[2].imshow(np.rot90(im[int(np.floor(im.shape[0]/2)),:,:]),alpha=0.5,cmap=cmaps[i])
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4()
    out = os.path.abspath('./debug_{}_{}.png'.format(timestr,uid))
    
    print('Saving {}'.format(out))
    fig.savefig(out)
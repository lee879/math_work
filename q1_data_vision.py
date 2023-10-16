import math

import matplotlib.pyplot as plt
import numpy as np
from data.util import sort_by_column

def data_vision(label,indx,sf,x_cut,color,Z,if_x_cut = False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    for num, (label,indx,s,xc,cr) in enumerate(zip(label[1:],indx,sf,x_cut,color)):
        i, j = indx
        smooth_factor = s
        x_0 = sort_by_column(Z,num+1)
        smoothed_y = np.convolve(x_0[:,0], np.ones(smooth_factor)/smooth_factor, mode='same')
        axs[i,j].plot(x_0[:,num+1],smoothed_y,color = cr)
        axs[i,j].set_xlabel(label)
        axs[i,j].set_ylabel("RVR_1A")
        if if_x_cut is True:
            axs[i,j].set_xlim(xc)
        axs[i,j].set_title("RVR_1A & " + label)
    plt.tight_layout()
    plt.show()

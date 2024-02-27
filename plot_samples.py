import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.gridspec import GridSpec

def process_images(data_dir):
    gt_data = {}
    sample_data = {}
    for f in os.listdir(data_dir):
        idx = "_".join(f.split('_')[0:2])
        if "mask" in f:
            gt_data[idx] = np.asarray(Image.open(os.path.join(data_dir, f)))
        else:
            if idx in sample_data:
                sample_data[idx].append(np.asarray(Image.open(os.path.join(data_dir, f))))
            else:
                sample_data[idx] = [np.asarray(Image.open(os.path.join(data_dir, f)))]


    return gt_data, sample_data

def plot_samples():
    data_dir = "rgb_samples_test"
    gt_data, samples = process_images(data_dir) 
    # print(gt_data.keys())
    # print(7/0)
    
    nrow = 6
    ncol = 6

    fig = plt.figure(figsize=(ncol*2, nrow*2)) 

    gs = GridSpec(nrow, ncol,
         wspace=0.02, hspace=0.02, 
         top=0.8, bottom=0.2, 
         left=0.2, right=0.8) 
    

    for (i,k) in enumerate(gt_data):
        gt_slice = gt_data[k]
        sample_1 = samples[k][0]
        sample_2 = samples[k][1]
        sample_3 = samples[k][2]
        sample_4 = samples[k][3]
        sample_5 = samples[k][4]



        for j in range(ncol):
            ax = fig.add_subplot(gs[i,j])
            ax.axis('off')
            if i == 0:
                if j == 0:
                    ax.set_title('Ground Truth')
                elif j == 1:
                    ax.set_title('Sample 1')
                elif j == 2:
                    ax.set_title('Sample 2')
                elif j == 3:
                    ax.set_title('Sample 3')
                elif j == 4:
                    ax.set_title('Sample 4')
                elif j == 5:
                    ax.set_title('Sample 5')
            if j == 0:
                ax.imshow(gt_slice)
            elif j == 1:
                ax.imshow(sample_1)
            elif j == 2:
                ax.imshow(sample_2)
            elif j == 3:
                ax.imshow(sample_3)
            elif j == 4:
                ax.imshow(sample_4)
            elif j == 5:
                ax.imshow(sample_5)

            
    plt.savefig('medseg_samples_test.jpg', bbox_inches='tight', pad_inches = 0.1) 



if __name__ == "__main__":
    plot_samples()
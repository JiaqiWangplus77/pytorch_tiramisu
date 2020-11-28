import numpy as np
import matplotlib.pyplot as plt


intact_road = [125,125,125]
applied_patch = [0,255,0]
pothole = [100,0,255]
inlaid_patch = [255,255,255]
open_joint = [204,0,255]
crack = [255,0,0]
#SignSymbol = [192,128,128]
#Fence = [64,64,128]
#Car = [64,0,128]
#Pedestrian = [64,64,0]
#Bicyclist = [0,128,192]
#Unlabelled = [0,0,0]

DSET_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]
label_colours = np.array([intact_road, applied_patch, pothole,
                          inlaid_patch, open_joint, crack])


def view_annotated(tensor, title,plot=True, class_num = 6):
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,class_num):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
        #plt.show()
        plt.title(title)
    return rgb

def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean
    return inp

def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

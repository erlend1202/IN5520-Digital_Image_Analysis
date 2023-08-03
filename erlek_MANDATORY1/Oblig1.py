from email.mime import image
from pyexpat import features
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#Function for visualizing a matrix
def visualize(array, bar=True, save=False, name="", only_im = False):
    if bar:
        plt.figure()
        plt.imshow(array)
        plt.colorbar()
    else:
        plt.figure(frameon=False)
        plt.imshow(array, cmap="gray")
    if save:
        if not only_im:
            plt.title(name)
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()

#Better function for saving images
def save_image_only(array, name=""):
    plt.figure()
    plt.axis('off')
    plt.imshow(array, cmap="gray")
    plt.savefig(name,bbox_inches='tight',transparent=True, pad_inches=0)

#Function for visualizing multiple images and saving them
def visualize_multiple(array, maxwidth=None, save=False, name=""):
    length = int(len(array)/2)
    plt.figure(figsize=(length*4, length*4))
    for i in range(len(array)):
        plt.subplot(length, length, i+1)
        plt.imshow(array[i])
        plt.colorbar()
    if save:
        plt.suptitle(name)
        plt.savefig(name)
    plt.show()
    

#Function for saving image
def save_image(array, name):
    new_image = Image.fromarray(array)
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')
    new_image.save(name)

#Function for visualizing histograms
def histogram_vis(array, multiple=False, save=False, name=""):
    if multiple:
        length = len(array)
        size = int(length/2)
        plt.figure(figsize=(length*2,length*2))
        plt.subplot(1, length-1, 1)
        plt.hist(array[0])
        for i in range(len(array)):
            plt.subplot(size,size, i+1)
            plt.hist(array[i])
        #plt.tight_layout()
        if save:
            plt.suptitle(name)
            plt.savefig(name)
        plt.show()

    else:
        plt.hist(array)
        if save:
            plt.title(name)
            plt.savefig(name)
        plt.show()

#Function for making GLCM 
def GLCM(im, dist=1, degree=90, L=16):
    #requantize
    if L != None:
        #im = Image.fromarray(im)
        #im = im.quantize(L, method=0)
        #im = np.array(im)
        im = np.floor(im * (L/255)) * (255/L)

    i_jump = 0
    j_jump = 0

    if degree == 0:
        j_jump = -1
    if degree == 45:
        i_jump = 1
        j_jump = -1
    if degree == 90:
        i_jump = 1
    if degree == 135:
        i_jump = 1
        j_jump = 1
    if degree == 180:
        j_jump = 1
    if degree == 225:
        i_jump = 1
        i_jump = -1
    if degree == 270:
        i_jump = -1
    if degree == 315:
        i_jump = -1
        j_jump = -1
    
    i_start = 1 if i_jump == -1 else 0
    i_end = -1 if i_jump == 1 else 0
    j_start = 1 if j_jump == -1 else 0
    j_end = -1 if j_jump == 1 else 0

    gray_levels = np.unique(im)
    map1 = {}
    map2 = {}
    for i in range(len(gray_levels)):
        map1[i] = gray_levels[i]
        map2[gray_levels[i]] = i

    num_gray_levels = len(gray_levels)
    
    Q = np.zeros((num_gray_levels,num_gray_levels))

    for i in range(i_start*dist, len(im) + i_end*dist):
        for j in range(j_start*dist, len(im[0]) + j_end*dist):
            val1 = im[i,j]
            val2 = im[i+i_jump*dist,j+j_jump*dist]
            Q[map2[val1],map2[val2]] += 1
    P = Q/(sum(sum(Q)))
    return Q,P

#Function for making Isotropic GLCM
def Isotropic_GLCM(im, dist, L):
    _,up = GLCM(im, dist=dist, degree = 0, L=L)
    _,up_right = GLCM(im, dist=dist, degree = 45, L=L)
    _,right = GLCM(im, dist=dist, degree = 90, L=L)
    _,down_right = GLCM(im, dist=dist, degree = 135, L=L)

    return (1/4)*(up + up_right + right + down_right)

#Functions bellow are implementations of IDM, Inertia and Cluster Shade
def IDM(P):
    sum = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            sum += P[i,j]/(1 + (i-j)**2)
    return sum

def Inertia(P):
    sum = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            sum += P[i,j]*((i-j)**2)
    return sum

def Cluster_Shade(P):
    my_i = 0
    my_j = 0
    temp_sum_i = 0

    for i in range(len(P)):
        temp_sum_i = 0
        for j in range(len(P[0])):
            my_j += j * P[i,j]
            temp_sum_i += P[i,j]
        my_i += i*temp_sum_i
    
    sum = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            sum += (i + j - my_i - my_j)**3 * P[i,j]

    return sum

#Function for creating a feature matrix
def feature_matrix(image, size, L=16 ,iso=False, function=IDM, d = 1, theta = 90):
    if L != None:
        image = Image.fromarray(image)
        image = image.quantize(L)
        image = np.array(image)
    out = np.ones_like(image, float)
    pad = int(size/2)
    image = np.pad(image, pad, 'reflect')
    
    for i in range(pad, len(image)-pad):
        for j in range(pad, len(image[0])-pad):
            if iso:
                P = Isotropic_GLCM(image[i-pad:i+pad,j-pad:j+pad], d, None)
            else:
                Q,P = GLCM(image[i-pad:i+pad,j-pad:j+pad], d, theta, L=None)
            feature = function(P)
            out[i-size,j-size] = feature
    return out


#function for making and visualizing all GLCM matrices.
def plot_all_GLCM(im_array, d, theta, save=False, name=""):
    Q_arr = []
    P_arr = []
    L = 16
    for image in im_array:
        Q,P = GLCM(image, d, theta, L)
        #P = Isotropic_GLCM(image,d,L)
        Q_arr.append(Q)
        P_arr.append(P)
    visualize_multiple(P_arr, save=save, name=name)






if __name__ == "__main__":
    mosaic1 = Image.open('mosaic1.png')
    mosaic1 = ImageOps.grayscale(mosaic1)
    mosaic_arr = np.array(mosaic1)


    im1 = mosaic_arr[0:256, 0:256]
    im2 = mosaic_arr[0:256, 256:512]
    im3 = mosaic_arr[256:512, 0:256]
    im4 = mosaic_arr[256:512, 256:512]
    im_array1 = [im1,im2,im3,im4]

    mosaic2 = Image.open('mosaic2.png')
    mosaic2 = ImageOps.grayscale(mosaic2)
    mosaic_arr2 = np.array(mosaic2)


    im12 = mosaic_arr2[0:256, 0:256]
    im22 = mosaic_arr2[0:256, 256:512]
    im32 = mosaic_arr2[256:512, 0:256]
    im42 = mosaic_arr2[256:512, 256:512]
    im_array2 = [im2,im22,im32,im42]

    #For histogram
    def save_hist():
        histogram_vis(mosaic_arr, save=True, name="hist_mosaic1")
        histogram_vis([im1,im2,im3,im4], True, save=True, name="hist_mosaic1_split")

        histogram_vis(mosaic_arr2, save=True, name="hist_mosaic2")
        histogram_vis([im12,im22,im32,im42], True, save=True, name="hist_mosaic2_split")


    #Finding d and theta to seperate the textures well
    def test_d_theta():
        d1 = 2
        theta1 = 180
        #plot_all_GLCM(im_array1, d1, theta1, save=True, name = "mosaic1_IsoGLCM")
        #d1 = 2 and theta1 = 180

        d2 = 4
        theta2 = 90
        #plot_all_GLCM(im_array2, d2, 90, save=True, name = "mosaic2_GLCM")
        #d2 = 2 and theta2 = 180
        return 0

    #Function for calculating feature matrix images
    def test_features():
        
        idm_image = feature_matrix(mosaic_arr, 31, iso=False, function=Cluster_Shade, d=2, theta=180)
        save_image_only(idm_image, "mosaic1_Cluster_Shade.png")
        print("done")

        idm_image = feature_matrix(mosaic_arr2, 31, iso=False, function=Cluster_Shade, d=2, theta=180)
        save_image_only(idm_image, "mosaic2_Cluster_Shade.png")
        print("done")

        idm_image = feature_matrix(mosaic_arr, 31, iso=False, function=IDM, d=2, theta=180)
        save_image_only(idm_image, "mosaic1_IDM.png")
        print("done")
        
        idm_image = feature_matrix(mosaic_arr2, 31, iso=False, function=IDM, d=2, theta=180)
        save_image_only(idm_image, "mosaic2_IDM.png")
        print("done")

        idm_image = feature_matrix(mosaic_arr, 31, iso=False, function=Inertia, d=2, theta=180)
        save_image_only(idm_image, "mosaic1_Inertia.png")
        print("done")

        idm_image = feature_matrix(mosaic_arr2, 31, iso=False, function=Inertia, d=2, theta=180)
        save_image_only(idm_image, "mosaic2_Inertia.png")
        print("done")

        return 0

    #Function for finding global thresholds
    def global_threshold():
        #Mosaic1
        ##Inertia: top right (0-40), bottom left ish (130,255)
        ##IDM: top right ish (85-150)
        ##Cluster Shade: couldnt get good result

        #Mosaic2
        ##Inertia: top right(0-75), left side (130,255), get bottom right by combining those 2
        ##IDM: top right(130-255), bottom left (0,60)
        ##Cluster Shade: bottom right (0,70), top right (125,255)

        feature_im = Image.open('mosaic1_IDM.png')
        feature_im = ImageOps.grayscale(feature_im)
        feature_arr = np.array(feature_im)
        threshold_up = 150
        threshold_down = 85
        for i in range(len(feature_arr)):
            for j in range(len(feature_arr[0])):
                if feature_arr[i,j] > threshold_down and feature_arr[i,j] < threshold_up:
                    feature_arr[i,j] = 0
                else:
                    feature_arr[i,j] = 255
        visualize(feature_arr, bar=False)
        save_image_only(feature_arr, "mosaic1_IDM_segment1")

    #save_hist()
    #test_d_theta()

    #d1 = 2
    #theta1 = 225
    #plot_all_GLCM(im_array1, d=2, theta=180, save=True, name = "mosaic1_GLCM")

    #test_features()
    #global_threshold()

    
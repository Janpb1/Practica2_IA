__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset
import numpy as np
from PIL import Image
from Kmeans import *
from KNN import *
import time as t

def Retrieval_by_color(list_img, color_labels, color_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if isinstance(color_question, str):
            for color in color_question:
                if color_labels[i] == color:
                    trobats.append(list_img[i])
                    break
        else:
            if color_labels[i] == color:
                    trobats.append(list_img[i])
    
    return trobats

def Retrieval_by_shape(list_img, shape_labels, shape_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if shape_labels[i] == shape_question:
            trobats.append(list_img[i])
    
    return trobats


def Retrieval_combined(list_img, shape_labels, color_labels, shape_question, color_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if shape_labels[i] == shape_question and color_labels[i] == color_question:
            trobats.append(list_img[i])
    
    return trobats

def Kmean_statistics(Kmeans_list, Kmax):
    WCD = np.zeros((len(Kmeans_list)))
    iter = np.zeros((len(Kmeans_list)))
    time = np.zeros((len(Kmeans_list)))
    
    for i in range(len(Kmeans_list)):
        WCD_list = []
        time_list = []
        Kmeans = Kmeans_list[i]
        for j in range(2, Kmax):
            Kmeans.K = j
            inici = t.process_time()
            Kmeans.fit()
            final = t.process_time()
            temps = inici - final 
            time_list.append(temps)
            Kmeans.within_class_distance()
            WCD_list.append(Kmeans.WCD)
        WCD[i] = WCD_list
        time[i] = time_list
        iter[i] = Kmeans.num_iter
        
    return WCD, time, iter
            
            
    
if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    test_imgs = test_imgs[:40]
    knn = KNN(train_imgs, train_class_labels)
    color_results = []
    label_results = knn.predict(test_imgs, 10)
    for image in test_imgs:
        km = KMeans(image, 3)
        km.fit()    
        colors = get_colors(np.array([list(km.centroids[0]), list(km.centroids[1]), list(km.centroids[2])]))
        color_results.append(colors)

    print("Retrieving grey flip-flops")
    grey_flip_flops = Retrieval_combined(test_imgs, label_results, color_results, "Flip Flops", ["Grey"])
    for image in grey_flip_flops:
        imageObj = Image.fromarray(image)
        imageObj.show()
    
    # You can start coding your functions here
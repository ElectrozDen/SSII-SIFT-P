import cv2
import os
import sys
import pickle
from ImagesManager import ImagesManager
from sklearn.cluster import KMeans


def print_class(manager, data, train):
    class0 = "Class 0:"
    class1 = "Class 1:"
    for index, lab in enumerate(data.labels_ if train else data):
        if lab == 0:
            class0 += (' ' + manager.files[index])
        else:
            class1 += (' ' + manager.files[index])
    print(class0)
    print(class1)
    return


detectors = {'sift': cv2.xfeatures2d.SIFT_create, 'surf': cv2.xfeatures2d.SURF_create, 'orb': cv2.ORB_create}

if sys.argv[1] == 'train':
    detector = None
    detector = detectors.get(sys.argv[2])()
    im = ImagesManager('./images', detector)
    print('---First Kmeans---')
    kmeans1 = KMeans(n_clusters=int(sys.argv[3]), random_state=0).fit(im.__get_all_descriptors__())
    im.__compute_bows__(kmeans1, int(sys.argv[3]), True)
    print('---Second Kmeans---')
    kmeans2 = KMeans(n_clusters=2, random_state=0).fit(im.__get_bows__())
    print('---Results---')
    print_class(im, kmeans2, True)
    print('---Saving Kmeans---')
    if os.path.exists('./kmeans_save') and os.path.isdir('./kmeans_save'):
        print('# kmeans_save folder already exist, files will be overwrite')
    else:
        os.mkdir('./kmeans_save')
    file = open('./kmeans_save/1.km', 'wb')
    pickle.dump(kmeans1, file)
    file.close()
    file = open('./kmeans_save/2.km', 'wb')
    pickle.dump(kmeans2, file)
    file.close()
else:
    file = open('./kmeans_save/1.km', 'rb')
    kmeans1 = pickle.load(file)
    file.close()
    detector = None
    detector = detectors.get(sys.argv[2])
    im = ImagesManager('./old_image', detector())
    print('---Predict with First Kmeans---')
    data = kmeans1.predict(im.__get_all_descriptors__())
    im.__compute_bows__(data, len(kmeans1.cluster_centers_), False)
    file = open('./kmeans_save/2.km', 'rb')
    kmeans2 = pickle.load(file)
    file.close()
    print('---Predict with Second Kmeans')
    data = kmeans2.predict(im.__get_bows__())
    print_class(im, data, False)

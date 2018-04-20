import cv2
import sys
import os
import pickle
from ImagesManager import ImagesManager
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def print_class(manager, data):
    class0 = "Class 0:"
    class1 = "Class 1:"
    for index, lab in enumerate(data):
        if lab == 0:
            class0 += (' ' + manager.files[index])
        else:
            class1 += (' ' + manager.files[index])
    print(class0)
    print(class1)
    return


detectors = {'sift': cv2.xfeatures2d.SIFT_create, 'surf': cv2.xfeatures2d.SURF_create, 'orb': cv2.ORB_create}
detector = detectors.get(sys.argv[2])

if sys.argv[1] == 'train':
    k = int(sys.argv[5])
    print('|||||||| First Class ||||||||')
    im_class0 = ImagesManager(sys.argv[3], detector())
    print('|||||||| Second Class ||||||||')
    im_class1 = ImagesManager(sys.argv[4], detector())
    data = []
    for d in im_class0.__get_all_descriptors__():
        data.append(d)
    for d in im_class1.__get_all_descriptors__():
        data.append(d)
    print('|||||||| Kmeans ||||||||')
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    print('|||||||| First Class ||||||||')
    im_class0.__compute_bows__(kmeans, k, True)
    print('|||||||| Second Class ||||||||')
    im_class1.__compute_bows__(kmeans, k, True, sum(nb_d for nb_d in im_class0.number_descriptors))

    logistic = LogisticRegression()
    labels = []
    for i in range(0, len(im_class0.files)):
        labels.append(0)
    for i in range(0, len(im_class1.files)):
        labels.append(1)
    bows = []
    for b in im_class0.__get_bows__():
        bows.append(b)
    for b in im_class1.__get_bows__():
        bows.append(b)
    print('|||||||| Logistic Regression ||||||||')
    logistic.fit(bows, labels)
    labelsPredicted = logistic.predict(bows)
    score = logistic.score(bows, labelsPredicted)
    print('### Score: ' + str(score))
    print('|||||||| Saving Logistic ||||||||')
    if os.path.exists('./logistic_save') and os.path.isdir('./logistic_save'):
        print('Save folder already exist, files will be overwrite')
    else:
        os.mkdir('./logistic_save')
    with open('./logistic_save/logr.lr', 'wb') as output:
        pickle.dump(logistic, output, pickle.HIGHEST_PROTOCOL)
    with open('./logistic_save/kmeans.k', 'wb') as output:
        pickle.dump(kmeans, output, pickle.HIGHEST_PROTOCOL)

else:
    kmeans = pickle.load(open('./logistic_save/kmeans.k', 'rb'))
    logistic = pickle.load(open('./logistic_save/logr.lr', 'rb'))

    im = ImagesManager(sys.argv[3], detector())
    kmeansP = kmeans.predict(im.__get_all_descriptors__())
    im.__compute_bows__(kmeansP, len(kmeans.cluster_centers_), False)
    labelsP = logistic.predict(im.__get_bows__())
    score = logistic.score(im.__get_bows__(), labelsP)
    print('### Score: ' + str(score))
    print('### Labels: ')
    print(labelsP)
    print_class(im, labelsP)

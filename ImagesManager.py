import cv2
import os
import shutil


class ImagesManager:

    def __init__(self, path, detector=None):
        self.detector = detector
        print('---Getting images---')
        self.files = os.listdir(path)
        print(self.files)
        print('---Reading images---')
        self.images = [cv2.imread(path + '/' + f) for f in self.files]
        print('---Getting gray of images---')
        self.grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
        self.keypoints = []
        self.bows = []
        self.descriptors = []
        self.number_descriptors = []
        self.path = path
        print('---Getting keypoints and descriptors')
        self.__compute()

    def __compute(self):
        if self.detector is None:
            return
        for index, g in enumerate(self.grays):
            k, d = self.detector.detectAndCompute(g, None)
            self.keypoints.append(k)
            self.descriptors.append(d)
            self.number_descriptors.append(len(d))
            print(self.files[index] + ': ' + str(len(d)) + ' descriptors')
        self.__generate_images()
        return

    def __generate_images(self):
        print('---Generating Images---')
        if os.path.exists('./output') and os.path.isdir('./output'):
            print('# Output folder detected, removing it')
            shutil.rmtree('./output')
        os.mkdir('./output')
        for index, gray in enumerate(self.grays):
            image = cv2.drawKeypoints(gray, self.keypoints[index], self.images[index],
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            path = 'keypoints_' + self.files[index]
            cv2.imwrite('./output/'+path, image)
            print('Created ' + path)
        return

    def __get_all_descriptors__(self):
        data = []
        for d in self.descriptors:
            for d_row in d:
                data.append(d_row)
        return data

    def __compute_bows__(self, data, nb_clusters, train, offset=0):
        print('---Getting bows---')
        i = 0 + offset
        for nb in self.number_descriptors:
            tmpBow = [0] * nb_clusters
            j = 0
            while j < nb:
                tmpBow[data.labels_[i] if train else data[i]] += 1
                j += 1
                i += 1
            self.bows.append(tmpBow)

    def __get_bows__(self):
        return self.bows;

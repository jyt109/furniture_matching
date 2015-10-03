__author__ = 'jeffreytang'

from mongo import Mongo
import pickle as pkl
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import os


class Comparison(object):

    def __init__(self, dbname, tabname):
        self.mongo_inst = Mongo(dbname, tabname)

    # Get descriptor numpy array by id from mongodb
    def query_descriptor_from_id(self, file_name):
        d = self.mongo_inst.find_one_by_id(file_name)
        return self._unpickle(d['des'])

    @staticmethod
    def _unpickle(binary):
        return pkl.loads(binary)

    @staticmethod
    def compare_des(des1, des2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # overall_squared_diff
        return np.mean([match.distance**2 for match in matches])

    def closest_k(self, query_des, k=10):
        if type(query_des) == str:
            query_des_arr = self.query_descriptor_from_id(query_des)
        elif type(query_des) == np.ndarray:
            query_des_arr = query_des
        else:
            raise TypeError('query_des must be a string (image file name) or numpy array (descriptor)')

        it = self.mongo_inst.tab.find()
        bottom_k_diff = np.empty(k)
        bottom_k_names = []
        i = 0
        for entry in it:
            des_arr = self._unpickle(entry['des'])
            name = entry['_id']

            mean_diff = self.compare_des(query_des_arr, des_arr)
            if i < k:
                bottom_k_diff[i] = mean_diff
                bottom_k_names.append(name)
            else:
                if mean_diff < np.min(bottom_k_diff):
                    min_ind = np.argmin(bottom_k_diff)
                    bottom_k_diff[min_ind] = mean_diff
                    bottom_k_names[min_ind] = name
            i += 1

        sorted_ind = np.argsort(bottom_k_diff)
        sorted_diff = bottom_k_diff[sorted_ind]
        sorted_name = np.array(bottom_k_names)[sorted_ind]
        return sorted_name, sorted_diff

    @staticmethod
    def plot_closest_10(dirname, sorted_names, sorted_diff):
        fig = plt.figure(1, (12., 5.))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.1)

        for i, name_diff in enumerate(zip(sorted_names, sorted_diff)):
            name, diff = name_diff
            image_arr = cv2.imread(os.path.join(dirname, name))
            grid[i].imshow(image_arr)  # The AxesGrid object work as a list of axes.
            grid[i].axes.get_xaxis().set_visible(False)
            grid[i].axes.get_yaxis().set_visible(False)
            label = mpatches.Patch(color='red', label=str(np.round(diff)))
            grid[i].legend(handles=[label], loc=4)
        plt.show()

if __name__ == '__main__':
    compare_inst = Comparison('object_matching', 'des')
    q1 = 'futon_BYV1686_1.jpg'
    names, diff = compare_inst.closest_k(q1)
    print names
    compare_inst.plot_closest_10('/Users/jeffreytang/furniture/object_matching/data/modern', names, diff)
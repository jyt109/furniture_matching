__author__ = 'jeffreytang'

import cv2
import os
from mongo import Mongo
from bson.binary import Binary
import pickle
import numpy as np


class KeyPointExtraction(object):

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def get_keypoints_from_image(self, file_path):
        file_full_path = os.path.join(self.dir_path, file_path)
        img = cv2.imread(file_full_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB()
        return orb.detectAndCompute(gray, None)

    def get_keypoints_from_dir(self, dbname=None, tabname=None, droptab=False):
        file_paths = [name for name in os.listdir(self.dir_path)
                      if name.endswith('.jpg') or name.endswith('.png')]

        mongo_inst = None
        if dbname is not None and tabname is not None:
            mongo_inst = Mongo(dbname, tabname)

        if mongo_inst is None:
            des_lst = []
            for file_path in file_paths:
                kp, des = self.get_keypoints_from_image(file_path)
                des_lst.append(des)
        else:
            if droptab:
                mongo_inst.tab.remove({})
            for file_path in file_paths:
                kp, des = self.get_keypoints_from_image(file_path)
                binary_des = Binary(pickle.dumps(des, protocol=2), subtype=128)
                entry = dict(_id=file_path, des=binary_des)
                mongo_inst.insert_without_warning(entry)

if __name__ == '__main__':
    # tests
    extractor = KeyPointExtraction('/Users/jeffreytang/furniture/object_matching/data/modern')
    kp_test, des_test = extractor.get_keypoints_from_image('futon_BYV1686_1.jpg')
    assert type(des_test) == np.ndarray
    extractor.get_keypoints_from_dir(dbname='object_matching', tabname='des', droptab=True)
    mongo = Mongo('object_matching', 'des')
    it = mongo.tab.find(dict(_id='futon_BYV3201_3.jpg'))
    test_entry = it.next()
    assert type(pickle.loads(test_entry['des'])) == np.ndarray

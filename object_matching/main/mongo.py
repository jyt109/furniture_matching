__author__ = 'jeffreytang'

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class Mongo(object):

    def __init__(self, db_name, tab_name):
        # assume local mongodb instance
        c = MongoClient()
        self.db = c[db_name]
        self.tab = self.db[tab_name]

    def insert_without_warning(self, entry):
        try:
            self.tab.insert(entry)
        except DuplicateKeyError as e:
            print e

    def find_one_by_id(self, entry_id):
        return self.tab.find_one(dict(_id=entry_id))

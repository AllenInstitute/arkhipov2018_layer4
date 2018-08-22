def is_sequence(obj):
    return (not hasattr(obj, "strip") and (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")))

class DataModel(object):
    ID_KEY = 'id'

    @staticmethod
    def new(type_name='dict', *args, **kwargs):
        if type_name == 'mongodb':
            from mongodb_data_model import MongoDBDataModel
            return MongoDBDataModel(*args, **kwargs)
        elif type_name == 'pandas':
            from pandas_data_model import PandasDataModel
            return PandasDataModel(*args, **kwargs)
        else:
            from dict_data_model import DictDataModel
            return DictDataModel(*args, **kwargs)

    def __init__(self):
        pass

    def select(self, collection, params=None, gids=None):
        raise NotImplementedError("Error: select not implemented")

    def select_ids(self, collection, params=None):
        raise NotImplementedError("Error: select_ids not implemented")

    def insert(self, collection, items):
        raise NotImplementedError("Error: insert not implemented")
    
    def count(self, collection, params=None):
        raise NotImplementedError("Error: count not implemented")

    def update(self, collection, array_data=None, update_data=None, gids=None):
        raise NotImplementedError("Error: update not implemented")

    def remove(self, collection, params=None, gids=None):
        raise NotImplementedError("Error: remove not implemented")



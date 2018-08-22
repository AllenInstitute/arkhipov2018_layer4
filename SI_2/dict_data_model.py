from data_model import DataModel
from data_model import is_sequence

from collections import OrderedDict

def get_dotted(obj, attr):
    attrs = attr.split('.')
    v = obj
    for a in attrs:
        v = v[a]
    return v

def set_dotted(obj, attr, val):
    attrs = attr.split('.')
    o = obj
    for a in attrs[0:-1]:
        so = o.get(a, None)

        if so is None or not isinstance(so, dict):
            o[a] = {}
            o = o[a]
        else:
            o = so

    o[attrs[-1]] = val
        
    return obj

class DictDataModel(DataModel):
    def __init__(self):
        super(DictDataModel, self).__init__()

        self.next_ids = {}
        self.data = {}

    def select(self, collection, params=None, gids=None): 
        if collection in self.data:
            items = self.data[collection]

            # if there are gids, filter the item list
            if gids is not None:
                out_items = ( items[gid] for gid in gids if gid in items )
            else:
                out_items = items.itervalues()
                
            # if there are params, filter some more
            if params is not None:
                for item in out_items:
                    skip = False

                    try:
                        for k,v in params.iteritems():
                            skip = (get_dotted(item,k) not in v) if is_sequence(v) else (get_dotted(item, k) != v)
                            if skip:
                                break
                    except:
                        skip = True

                    if not skip:
                        yield item
            else:
                for item in out_items:
                    yield item

    def select_ids(self, collection, params=None):
        return ( item[self.ID_KEY] for item in self.select(collection,params) )

    def insert(self, collection, items):
        idk = self.ID_KEY

        od = self.data.get(collection, None)

        if od is None:
            od = OrderedDict()
            self.data[collection] = od

        for item in items:
            # clobber the id field of the item (or assign it if it didn't already exist)
            item[idk] = self.next_id(collection)
            od.update({ item[idk]: item })

    def next_id(self, collection, value=None):
        next_id = self.next_ids.get(collection, 0)
        self.next_ids[collection] = next_id+1
        return next_id

    def count(self, collection, params=None):
        try:
            if params is None:
                return len(self.data[collection])
            else:
                return len(self.select(collection, params))
        except:
            return 0

    def update(self, collection, array_data=None, update_data=None, gids=None):
        try:
            d = self.data[collection]
        except:
            return None

        if update_data is not None:
            if gids is None:
                for item in d.itervalues():
                    item.update(update_data)
            else:
                for gid in gids:
                    d[gid].update(update_data)

        if array_data is not None:
            if gids is None:
                for key, values in array_data.iteritems():
                    i = 0
                    for item in d.itervalues():
                        item[key] = values[i]
                        i += 1
            else:
                for key, values in array_data.iteritems():
                    for i, gid in enumerate(gids):
                        d[gid][key] = values[i]

    def remove(self, collection, params=None, gids=None):
        if params is None and gids is None:
            if collection in self.data:
                del self.data[collection]
        else:
            d = self.data[collection]
            for item in self.select(collection, params, gids):
                del d[item[self.ID_KEY]]
            



import json
import numpy as np
import scipy.sparse
import os
import h5py
from scipy.sparse import csr_matrix

class ItemListHandler( object ):
    NAME = '__item_list__'

    @staticmethod
    def item_list_to_property_arrays(item_list):
        num_items = len(item_list)

        property_arrays = {}
        bad_properties = []

        # first pass, initialize property arrays
        for item in item_list:
            for k,v in item.iteritems():
                if k not in property_arrays:
                    a = ItemListHandler.initialize_array_from_value(v, num_items)
                    property_arrays[k] = a
                    if a is None:
                        bad_properties.append(k)

        # delete the bad properties
        for bad_property in bad_properties:
            del property_arrays[bad_property]

        # now fill in the property arrays
        for i,item in enumerate(item_list):
            for k,a in property_arrays.iteritems():
                v = item.get(k, None)
                if v is not None:
                    a[i] = v

        return property_arrays

    @staticmethod
    def property_arrays_to_item_list(property_arrays):
        item_list = []
        
        keys = property_arrays.keys()
        n = property_arrays[keys[0]].shape[0]

        for i in xrange(n):
            item_list.append(dict(( k,property_arrays[k][i]) for k in keys ))

        return item_list

    @staticmethod
    def initialize_array_from_value(value, N):
        item_type = type(value)
            
        if item_type in PRIMITIVE_TYPES:
            return np.zeros(N, dtype=item_type)
        elif item_type == str:
            return np.zeros(N, dtype='S256')
        elif item_type == np.ndarray:
            return np.zeros((N,value.size), dtype=value.dtype)
        elif item_type in np.sctypes['float'] or item_type in np.sctypes['int'] or item_type in np.sctypes['uint']:
            return np.zeros((N,), dtype=type(value))
        else:
            return None

    def write(self, h5_file, obj, key):
        item_list = obj[key]
        property_arrays = ItemListHandler.item_list_to_property_arrays(item_list)

        for property_name, values in property_arrays.iteritems():
            h5_file[key + '/' + property_name] = values

    def read(self, h5_file, obj, key):
        property_arrays = {}
        
        hg = h5_file[key]
        property_names = hg.keys()
        for property_name in property_names:
            property_arrays[property_name] = hg[property_name].value
            
        return ItemListHandler.property_arrays_to_item_list(property_arrays)
        #return property_arrays
        
class ConnectionSetHandler( object ):        
    NAME = '__connection_set__'

    @staticmethod
    def connection_set_to_property_arrays(connection_set):
        N = 0
        
        for sid, tid, v in connection_set:
            N = max(N, sid, tid)
            
        m = scipy.sparse.dok_matrix((N+1,N+1), dtype=int)

        ct = 0
        connection_list = []
        for sid, tid, v in connection_set:
            m[sid, tid] = ct
            connection_list.append(v)
            ct += 1

        m = m.tocsr()

        return m, ItemListHandler.item_list_to_property_arrays(connection_list)
        
    def write(self, h5_file, obj, key):
        connection_set = obj[key]

        matrix, property_arrays = ConnectionSetHandler.connection_set_to_property_arrays(connection_set)

        hg = h5_file.create_group(key)
        
        mg = hg.create_group('matrix')

        mg['shape'] = matrix.shape
        mg['indices'] = matrix.indices
        mg['indptr'] = matrix.indptr
        mg['data'] = matrix.data

        for property_name, values in property_arrays.iteritems():
            h5_file[key + '/' + property_name] = values

    def read(self, h5_file, obj, key):
        hg = h5_file[key]

        property_arrays = {}

        for hkey in hg.keys():
            if hkey == 'matrix':
                matrix_shape = hg['matrix/shape'].value
                matrix_indices = hg['matrix/indices'].value
                matrix_indptr = hg['matrix/indptr'].value
                matrix_data = hg['matrix/data'].value

                matrix = csr_matrix((matrix_data,
                                     matrix_indices,
                                     matrix_indptr),
                                    shape=matrix_shape)

            else:
                property_arrays[hkey] = hg[hkey].value

        connection_list = ItemListHandler.property_arrays_to_item_list(property_arrays)
        
        
        return property_arrays


PRIMITIVE_TYPES = (int, bool, float, int, long)

TYPE_HANDLERS = {
    ItemListHandler.NAME: ItemListHandler(),
    ConnectionSetHandler.NAME: ConnectionSetHandler()
}

def write(obj, file_name, key_handler_types=None):
    obj = dict(obj)

    if key_handler_types is None:
        write_plain_json(obj, file_name)
        return

    base, ext = os.path.splitext(file_name)
    h5_file_name = base + '.h5'

    hf = h5py.File(h5_file_name, 'w')
    
    for key, handler_type in key_handler_types.iteritems():
        handler = TYPE_HANDLERS[handler_type]

        handler.write(hf, obj, key)
        
        obj[key] = {
            '__nio_type__': handler.NAME,
            '__h5_file__': h5_file_name
        }

    hf.close()

    write_plain_json(obj, file_name)

def read(file_name):
    obj = json.loads(open(file_name).read())

    keys_to_replace = []
    for key, value in obj.iteritems():
        if isinstance(value, object) and '__nio_type__' in value:
            keys_to_replace.append(key)

    if len(keys_to_replace) == 0:
        return obj

    for key in keys_to_replace:
        value = obj[key]

        nio_type = value['__nio_type__']
        h5_file_name = value['__h5_file__']

        h5f = h5py.File(h5_file_name, 'r')
        
        type_handler = TYPE_HANDLERS[nio_type]
        data = type_handler.read(h5f, value, key)
        
        h5f.close()

        obj[key] = data

    return obj


def write_plain_json(obj, file_name):
    with open(file_name, 'wb') as f:
        f.write(json.dumps(obj, indent=2))
                  

                

        

import scipy.sparse
import numpy as np
import itertools

class ConnectionSet( object ):
    SOURCE_KEY = 'source_id'
    TARGET_KEY = 'target_id'

    ''' 
    The ConnectionSet represents connections as a dictionary of dictionaries
    (e.g. d[source][target] = value). It behaves much like a normal dictionary, 
    but has some extra convenience methods. 
    '''
    def __init__(self, data_model, collection_name='connections', matrix=None):
        self.data_model = data_model
        self.collection_name = collection_name

        self.matrix = {}
        if matrix:
            self.set_matrix(matrix)

    def set_matrix(self, matrix):
        coo = matrix.tocoo(False)
        self.matrix = {}
        self.matrix.update(itertools.izip(itertools.izip(coo.row,coo.col),coo.data))

    def add_many(self, connections):
        for sid, tid, connection in connections:
            self.add_one(sid, tid, connection)

    def add_list(self, connections):
        for connection in connections:
            self.add_one(connection[self.SOURCE_KEY], connection[self.TARGET_KEY], connection)

    def add_one(self, sid, tid, params):
        if self.SOURCE_KEY not in params:
            params[self.SOURCE_KEY] = sid
        if self.TARGET_KEY not in params:
            params[self.TARGET_KEY] = tid

        if (sid,tid) in self.matrix:
            r = self.matrix[sid,tid]
            self.data_model.update(self.collection_name, update_data=params, gids=[r])
        else:
            self.matrix[sid,tid] = len(self.matrix)
            self.data_model.insert(self.collection_name, [params])

    def remove_one(self, sid, tid):
        gid = self.matrix.pop((sid,tid))
        self.data_model.remove(self.collection_name, gids=[gid])

    def remove_many(self, connections):
        gids_to_remove = [ self.matrix.pop((sid,tid)) for sid, tid, connection in connections ]
        self.data_model.remove(self.collection_name, gids=gids_to_remove)

    def __iter__(self):
        for item in self.data_model.select(self.collection_name):
            yield item[self.SOURCE_KEY], item[self.TARGET_KEY], item

    def __getitem__(self, index):
        row_id = self.matrix[index]
        return self.data_model.select(self.collection_name, gids=[row_id]).next()

    def get_many(self, keys):
        row_ids = []
        for key in keys:
            try:
                row_ids.append(self.matrix[key])
            except:
                pass

        if len(row_ids) > 0:
            return ( (n[self.SOURCE_KEY], n[self.TARGET_KEY], n) for n in self.data_model.select(self.collection_name, gids=row_ids) )
        else:
            return ()

            

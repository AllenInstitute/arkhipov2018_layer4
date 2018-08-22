from data_model import DataModel
from data_model import is_sequence
import pandas
from collections import OrderedDict

class PandasDataModel(DataModel):

    def __init__(self):
        super(PandasDataModel, self).__init__()

        self.data_frames = {}

    def select(self, collection, params=None, gids=None, only=None):
        try:
            df = self.data_frames[collection]
        except:
            return None

        if params is not None:
            for k,v in params.iteritems():
                if is_sequence(v):
                    df = df[df[k].isin(v)]
                else:
                    df = df[df[k] == v]

        if gids is not None:
            df = df[df[self.ID_KEY].isin(gids)]


        if only is not None:
            data = list(df[only])
        else:
            data = df.to_dict(outtype='records')
            
        return ( n for n in data )

    def select_ids(self, collection, params=None):
        return self.select(collection, params, only=self.ID_KEY)
        
    def insert(self, collection, items):
        df = self.data_frames.get(collection, None)
        
        if df is None:
            next_id = 0
        else:
            next_id = len(df)

        for item in items:
            if self.ID_KEY not in item:
                item[self.ID_KEY] = next_id
                next_id += 1

        if df is None:
            df = pandas.DataFrame(items)
            self.data_frames[collection] = df
        else:
            df = self.data_frames[collection].append(items, ignore_index=True)
            self.data_frames[collection] = df

    def count(self, collection, params=None):
        try:
            df = self.data_frames[collection]
        except:
            return 0

        return len(df.index)


    def update(self, collection, array_data=None, update_data=None, gids=None):
        try:
            df = self.data_frames[collection]
        except:
            return None

        if gids:
            if array_data:
                for key, values in array_data.iteritems():
                    if key not in df.columns:
                        df[key] = 0
                        
                    df[key].iloc[gids] = list(values)
            if update_data:
                for key,v in update_data.iteritems():
                    df[key].iloc[gids] = v
        else:
            if array_data:
                for key, values in array_data.iteritems():
                    df[key] = list(values)

            if update_data:
                for k,v in update_data.iteritems():
                    df[k] = v

    def remove(self, collection, params=None, gids=None):
        if params is None and gids is None:
            try:
                del self.data_frames[collection]
            except:
                pass
            return 
        else:
            df = self.data_frames[collection]
            gids = [ item[self.ID_KEY] for item in self.select(collection, params,gids) ]
            df.drop(df.index[gids], inplace=True)

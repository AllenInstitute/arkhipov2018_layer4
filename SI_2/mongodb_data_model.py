import pymongo

from data_model import DataModel

class MongoDBDataModel(DataModel):

    def __init__(self, database_name, database_url=None, database_client=None, reinitialize_database=True):
        super(MongoDBDataModel, self).__init__()

        self.client = database_client

        if self.client is None:
            assert database_url is not None, "Error: database requires a mongodb URL if no client is given"
            self.client = pymongo.MongoClient(database_url)

        if reinitialize_database:
            self.client.drop_database(database_name)

        self.database = self.client[database_name]

    def select(self, collection, params=None, ids=None, fields=None):
        try:
            self.assert_collection_exists(collection)
        except Exception:
            return []

        if params is None:
            params = {}

        if ids is not None:
            params[self.ID_KEY] = { '$in': ids }

        qfields = {
            '_id': False
        }

        if fields is not None:
            qfields.update(fields)

        return list(self.database[collection].find(params, fields=qfields))

    def select_ids(self, collection, params=None):
        return [ n[self.ID_KEY] for n in self.select(collection, params, fields={self.ID_KEY:True}) ]

    def update(self, collection, array_data=None, update_data=None, gids=None):
        self.assert_collection_exists(collection)

        collection = self.database[collection]

        print "building update"
        bulk = collection.initialize_unordered_bulk_op()

        if gids:
            if update_data:
                bulk.find({self.ID_KEY: { '$in': gids }}).update(update_data)

            if array_data:
                for i,gid in enumerate(gids):
                    for key, values in array_data.iteritems():
                        bulk.find({self.ID_KEY: gid}).update_one({'$set': { key: values[i] }})
        else:
            if update_data:
                bulk.find({}).update(update_data)

            if array_data:
                for i in xrange(collection.count()):
                    for key, values in array_data.iteritems():
                        bulk.find({self.ID_KEY: i}).update_one({'$set': { key: values[i] }})

        print "executing"

        bulk.execute()

        print "done"

    def insert(self, collection, items):
        if len(items) > 0:
            collection = self.database[collection]
            collection.insert(items)

    def count(self, collection, params=None):
        try:
            self.assert_collection_exists(collection)

            if params is not None:
                return self.database[collection].find(params).count()
            else:
                return self.database[collection].count()
        except Exception:
            return 0

    def assert_collection_exists(self, collection):
        assert collection in self.database.collection_names(), "%s does not exist in database" % (collection)

    def remove(self, collection, params=None, gids=None):
        if params is None:
            params = {}

        collection = self.database[collection]
        collection.remove(params)
        


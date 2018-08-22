import nio as nio
from data_model import DataModel
from connection_set import ConnectionSet
import connector as connector
import numpy as np
import random
import itertools

class Network( object ):
    '''
    A simple network class that organizes nodes, node types, and connections.
    '''

    def __init__(self, data_model=None):
        if data_model is None:
            self.data_model = DataModel.new()
        else:
            self.data_model = data_model

        self.connection_set = ConnectionSet(self.data_model)
        self.node_types = {}

    def nodes(self, params=None, gids=None):
        return self.data_model.select('nodes', params, gids)

    def node_ids(self, params=None):
        return self.data_model.select_ids('nodes', params)

    def node_count(self, params=None):
        return self.data_model.count('nodes', params)

    def update_nodes(self, array_data=None, update_data=None, gids=None):
        self.data_model.update('nodes', array_data, update_data, gids)

    def seed(self, v):
        '''
        take care of seeing both numpy and the python rngs
        '''
        np.random.seed(v)
        random.seed(v)

    
    def copy_nodes(self, node_params=None, node_ids=None, update_node_params=None, update_connection_params=None):

        if update_connection_params is None:
            update_connection_params = {}

        nid_key = DataModel.ID_KEY
        sid_key = ConnectionSet.SOURCE_KEY
        tid_key = ConnectionSet.TARGET_KEY

        num_nodes = self.data_model.count('nodes')
        nodes = list(self.nodes(node_params, node_ids))
        new_node_ids = dict((n[nid_key], i+num_nodes) for i,n in enumerate(nodes) )
#         new_node_ids = { n[nid_key]: i+num_nodes for i,n in enumerate(nodes) }

        # copy the connections
        for sid, tid, c in self.connection_set:
            new_sid = new_node_ids.get(sid, None)
            new_tid = new_node_ids.get(tid, None)

            if new_sid or new_tid:
                nc = dict(c)
                
                if new_sid:
                    nc[sid_key] = new_sid
                if new_tid:
                    nc[tid_key] = new_tid

                del nc[nid_key]

                if update_connection_params:
                    nc.update(update_connection_params)
                    
                self.connection_set.add_one(nc[sid_key], nc[tid_key], nc)

        num_new_nodes = len(nodes)

        # update the ids
        if update_node_params is None:
            update_node_params = {}

        new_nodes = []
        for i in xrange(num_new_nodes):
            new_node = dict(nodes[i])
            del new_node[nid_key]
            new_node.update(update_node_params)
            new_nodes.append(new_node)

        self.data_model.insert('nodes', new_nodes)

        return range(num_nodes, num_nodes+num_new_nodes)

    def copy_type(self, from_name, to_name, params=None):
        ''' 
        Copy the properties of one type to another, updating parameters as necessary.
        from_name: source node type name
        to_name: target (new) node type name 
        params: node type parameters for the new type
        '''
        if params is None:
            params = {}

        # make sure that the from type exists.
        from_type = self.node_types.get(from_name, None)
        assert from_type is not None, "Error: %s is not a known node type" % (from_name)
        
        # make sure that the new type does NOT exist.
        to_type = self.node_types.get(to_name, None)
        assert to_type is None, "Error: %s already exists" % (to_name)

        # update the properties of the new type
        to_type = dict(**from_type)
        to_type.update(params)

        # store the new type in the type dictionary
        self.nodes_types[to_type] = to_type

    def create_nodes(self, type_name, N=1, categories=None, type_params=None, instance_params=None):
        ''' 
        Create N instances of a particular node type, giving them optional additional specific parameters.
        If the type is not already known, register it.
        type_name: name of the new node type.
        N: how many instances to create
        params: update parameters for the given type.
        '''

        if type_params is None:
            type_params = {}

        if categories is None:
            categories = {}

        # get the node type parameters
        node_type = self.node_types.get(type_name, None)

        # if the type doesn't exist, register it
        if node_type is None:
            self.node_types[type_name] = type_params
            node_type = type_params
        else:
            # the type does exist -- update its parameters.
            node_type.update(type_params)

        num_nodes = self.data_model.count('nodes')

        categories['type'] = [type_name]
        category_keys = categories.keys()
        category_values_list = list(itertools.product(*categories.values()))

        # make the new nodes
        category_nodes = [ dict(zip(category_keys, category_values)) for category_values in category_values_list ] 
        new_nodes = [ dict(d) for d in category_nodes for i in xrange(N) ]

        num_new_nodes = len(new_nodes)

        if instance_params:
            for n in new_nodes:
                n.update(instance_params)

        self.data_model.insert('nodes', new_nodes)

        return range(num_nodes, num_nodes + len(new_nodes))

    def remove_nodes(self, params=None, gids=None):
        self.remove_connections(node_params=params, node_ids=gids)
        self.data_model.remove('nodes', params, gids)

    def remove_connections(self, source_params=None, source_gids=None,
                           target_params=None, target_gids=None,
                           node_params=None, node_ids=None):

        connections = self.connections(source_params, source_gids, 
                                       target_params, target_gids,
                                       node_params, node_ids)

        self.connection_set.remove_many(connections)

    def write(self, file_name):
        '''
        Write the network properties to a JSON file.
        '''
        data = self.to_json()
        #nio.write(data, file_name, key_handler_types={'nodes': '__item_list__', 'connections': '__connection_set__'})
        nio.write(data, file_name, key_handler_types={'nodes': '__item_list__', 'connections': '__item_list__'})

    def to_json(self):
        return {
            'node_types': self.node_types.values(),
            'nodes': list(self.nodes()), 
            'connections': list( v for sid, tid, v in self.connection_set )
        }

    def connections(self, source_params=None, source_ids=None, 
                    target_params=None, target_ids=None,
                    node_params=None, node_ids=None,
                    connection_params=None, connection_ids=None):

        if node_params is not None or node_ids is not None:
            return self.connections_undirected(node_params, node_ids)
        elif source_params is not None or target_params is not None or source_ids is not None or target_ids is not None:
            return self.connections_directed(source_params, source_ids, target_params, target_ids)
        elif connection_params is not None or connection_ids is not None:
            return ( (c[ConnectionSet.SOURCE_KEY], c[ConnectionSet.TARGET_KEY], c) for c  in self.data_model.select('connections', connection_params, connection_ids) )
        else:
            return self.connection_set
    
    def connections_directed(self, source_params=None, source_ids=None, 
                             target_params=None, target_ids=None):
        if source_params is not None:
            sids = set(self.node_ids(source_params))
        else:
            sids = set()

        if target_params is not None:
            tids = set(self.node_ids(target_params))
        else:
            tids = set()

        if source_ids is not None:
            sids.update(set(source_ids))
            
        if target_ids is not None:
            tids.update(set(target_ids))

        return self.connection_set.get_many(itertools.product(sids,tids))

        #for sid,tid in itertools.product(sids,tids):
        #    try:
        #        yield sid, tid, self.connection_set[sid,tid]
        #    except Exception, e:
        #        # fail silently if the connection does not exist
        #        pass

    def connections_undirected(self, node_params=None, node_ids=None):
        if node_params is not None:
            nids = set(self.node_ids(node_params))
        else:
            nids = set()

        if node_ids is not None:
            nids.update(set(node_ids))

        for sid, tid, c in self.connection_set:
            if sid in nids or tid in nids:
                yield sid, tid, c
        
    def connect(self, source_gids=None, target_gids=None, 
                source_params=None, target_params=None,
                join_on=None,
                connection_type='all_to_all', connection_type_params=None, 
                weight_function=None, weight_function_params=None):

        cnct = self.connector(source_gids, target_gids, source_params, target_params,
                              join_on, connection_type, connection_type_params,
                              weight_function, weight_function_params)

        connections = ( (src[DataModel.ID_KEY], tgt[DataModel.ID_KEY], v) for src, tgt, v in cnct )
        self.connection_set.add_many(connections)        


    def connector(self, source_gids=None, target_gids=None, 
                source_params=None, target_params=None,
                join_on=None,
                connection_type='all_to_all', connection_type_params=None, 
                weight_function=None, weight_function_params=None):

        if connection_type_params is None:
            connection_type_params = {}

        source_nodes = list(self.nodes(source_params, source_gids))
        target_nodes = list(self.nodes(target_params, target_gids))

        cnct = connector.create(source_nodes,
                                target_nodes,
                                connector_type=connection_type,
                                **connection_type_params)
        
        cnct.set_weight_function(weight_function, weight_function_params)
        cnct.set_join_keys(join_on)

        return cnct

    @staticmethod
    def from_json(file_name, data_model_type='dict'):
        '''
        Import a network from a JSON file using the nio library.
        '''
        data_model = DataModel.new(data_model_type)
        network = Network(data_model)
        data = nio.read(file_name)

        nodes = data.get('nodes', None)
        if nodes is not None:
            network.data_model.insert('nodes', nodes)

        types = data.get('types', None)
        if types is not None:
            network.types = types

        connections = data.get('connections', None)
        if connections is not None:
            network.connection_set.add_list(connections)

        return network




        

    

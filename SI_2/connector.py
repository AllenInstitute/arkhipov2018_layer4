import random
import itertools
import functools
import json
import bisect

class Connector( object ):
    def __init__(self, sources, targets):
        self.sources = list(sources)
        self.targets = list(targets)

        self.weight_function = lambda x,y: {}
        self.join_keys = []

    def set_weight_function(self, weight_function, weight_function_params):
        if weight_function is None:
            weight_function = lambda x,y: {}
            
        if weight_function_params is None:
            weight_function_params = {}

        self.weight_function = functools.partial(weight_function, **weight_function_params)

    def set_join_keys(self, join_keys):
        self.join_keys = join_keys or []
        
    def process_connection(self, source, target):
        for join_key in self.join_keys:
            if source[join_key] != target[join_key]:
                return None
                    
        return self.weight_function(source, target)

    def __iter__(self):
        counter = 0
        N_iterations = len(self.sources) * len(self.targets)
        for source, target in itertools.product(self.sources, self.targets):

            if (counter % 100000 == 0):
              print 'Establishing connections; progress %.3f percent.' % (100.0 * counter / N_iterations)
            counter += 1

            v = self.process_connection(source, target)
            if v is not None:
                yield source, target, v

class FixedInDegreeConnector( Connector ):
    '''
    A Connector subclass that yields a connection set for which all supplied targets have N incoming edges.
    '''

    def __init__(self, sources, targets, N):
        super(FixedInDegreeConnector, self).__init__(sources, targets)
        self.N = N
    
    def __iter_(self):
        sources = random.shuffle(self.sources)
        for target in self.targets:
            n = 0
            for source in sources:
                v = self.process_connection(source, target)
                if v is not None:
                    yield source, target, v
                    n+=1

                if n >= self.N:
                    break

class RangeInDegreeConnector( Connector ):
    '''
    A Connector subclass that yields a connection set for which all supplied targets have and random number of incoming edges.
    '''

    def __init__(self, sources, targets, rrange):
        super(RangeInDegreeConnector, self).__init__(sources, targets)
        self.rrange = rrange
    
    def __iter_(self):
        sources = random.shuffle(self.sources)
        for target in self.targets:
            n = 0
            N = random.randrange(self.rrange[0], self.rrange[1])
            for source in sources:
                v = self.process_connection(source, target)
                if v is not None:
                    yield source, target, v
                    n+=1

                if n >= N:
                    break
            
class FixedOutDegreeConnector( Connector ):
    '''
    A Connector subclass that yields a connection set for which all supplied sources have N outgoing edges.
    '''

    def __init__(self, sources, targets, N):
        super(FixedOutDegreeConnector, self).__init__(targets, sources)
        self.N = N

    def __iter_(self):
        targets = random.shuffle(self.targets)
        for source in self.sources:
            n = 0
            for target in targets:
                v = self.process_connection(source, target)
                if v is not None:
                    yield source, target, v
                    n+=1

                if n >= self.N:
                    break

class RangeOutDegreeConnector( Connector ):
    '''
    A Connector subclass that yields a connection set for which all supplied sources have a random number of outgoing edges.
    '''

    def __init__(self, sources, targets, rrange):
        super(RangeOutDegreeConnector, self).__init__(targets, sources)
        self.rrange = rrange

    def __iter_(self):
        targets = random.shuffle(self.targets)
        for source in self.sources:
            N = random.randrange(self.rrange[0], self.rrange[1])
            n = 0
            for target in targets:
                v = self.process_connection(source, target)
                if v is not None:
                    yield source, target, v
                    n+=1

                if n >= N:
                    break

class PairwiseRandomConnector( Connector ):
    '''
    A Connector subclass that gives all source-target pairs a certain probability of existing.
    '''

    def __init__(self, sources, targets, p):
        super(RandomConnector, self).__init__(sources, targets)
        self.p = p

    def __iter__(self):
        for source, target in itertools.product(self.sources, self.targets):
            if random.random() < self.p:
                v = self.process_connection(source, target)
                if v is not None:
                    yield source, target, v

CONNECTOR_TYPE_MAP = {}

def register_connector(cls, name):
    CONNECTOR_TYPE_MAP[name] = cls

def create(sources, targets, connector_type=None, *args, **kwargs):
    connector_class = CONNECTOR_TYPE_MAP.get(connector_type, None)

    if connector_class is not None:
        return connector_class(sources, targets, *args, **kwargs)
    else:
        raise Exception("Error: %s is not a connector type" % (connector_type))

register_connector(Connector, 'all_to_all')
register_connector(Connector, None)
register_connector(FixedInDegreeConnector, 'fixed_in_degree')
register_connector(FixedOutDegreeConnector, 'fixed_in_degree')
register_connector(RangeInDegreeConnector, 'range_in_degree')
register_connector(RangeOutDegreeConnector, 'range_out_degree')
register_connector(PairwiseRandomConnector, 'pairwise_random')

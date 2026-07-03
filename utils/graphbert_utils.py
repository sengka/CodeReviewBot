import numpy as np
from node2vec import Node2Vec
 
def graph_to_encoding(graph, dimensions=64):
    if graph.number_of_nodes() == 0:
        return np.zeros(dimensions)
    n2v = Node2Vec(graph, dimensions=dimensions, walk_length=10,
                   num_walks=20, workers=1, quiet=True)
    m = n2v.fit(window=5, min_count=1)
    return np.mean([m.wv[str(n)] for n in graph.nodes()], axis=0)

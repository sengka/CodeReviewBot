import networkx as nx
 
def ast_to_graph(node, graph=None, parent_id=None, node_id=[0]):
    if graph is None:
        graph = nx.DiGraph()
        node_id[0] = 0
    current_id = node_id[0]
    node_id[0] += 1
    graph.add_node(current_id, type=node.type,
                   text=node.text.decode("utf-8") if node.text else "")
    if parent_id is not None:
        graph.add_edge(parent_id, current_id)
    for child in node.children:
        ast_to_graph(child, graph, current_id, node_id)
    return graph

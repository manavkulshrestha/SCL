def print_dep(graph, node_ids, name):
    """
    graph is the dependence graph,
    node_ids is a list of object ids,
    name is a function which returns name from object id
    """
    for i, r in enumerate(graph):
        for j, c in enumerate(r):
            if c:
                nidi = int(node_ids[i])
                nidj = int(node_ids[j])
                print(f'{name(nidi)}[{nidi}] depends on {name(nidj)}[{nidj}]')
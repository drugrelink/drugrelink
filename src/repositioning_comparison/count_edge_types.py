# -*- coding: utf-8 -*-
'''Count the number of edge types given a graph'''

import networkx as nx
from collections import Counter

def count(graph):
    data = graph.edges.data()
    type_list=[]
    for i in data:
        type_list.append(i[-1]["type"])
    n = Counter(type_list).values()
    return n

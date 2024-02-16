from itertools import permutations
from math import perm
import pandas as pd
import re


try:
    from .core import Cartogram
except ImportError:
    from core import Cartogram

import numpy as np
from collections import defaultdict

class Graphs(Cartogram):
    ORDER = [
        0, 1, 2, 3, 
        7, 9, 11, 15, 
        19, 18, 17, 16, 
        12, 10, 8, 4
    ]
    
    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        nodes: list | None = None, 
        prefix: str = ""
    ):
        super().__init__(df=df)
        if nodes is None:
            self.nodes = self.CORE_PARTS["default"]["ALL_CELLS"]
        else:
            self.nodes = nodes
        
        if prefix != "":
            self.nodes = [i + prefix for i in self.nodes]

        self.prefix = prefix
        self.graph = dict()  #* ordered pairs
        self.visited = []
        
    def _make_graph(self):
        #* a rule is closest nodes are connected
        for n, i in enumerate(self.nodes):
            self.graph[i] = self._find_pairs(i)

        return

    def _find_pairs(self, node):
        col, row = list(
            map(
                lambda x: int(x), 
                re.findall(r"\d+", node)
            )
        )
        v = []

        for i in range(1,3):
            if i % 2 == 0:
                v.append(self._make_node(col - 1, row))
            else:
                v.append(self._make_node(col + 1, row))

        for i in range(1,3):
            if i % 2 == 0:
                v.append(self._make_node(col, row - 1))
            else:
                v.append(self._make_node(col, row + 1))

        return list(set(self.nodes).intersection(set(v)))
            
    
    def _make_node(self, col: int, row: int):
        return f"{col}-{row}{self.prefix}"

    def make_walk_route(self, key:str):
        default_order =  list(np.asarray(list(self.graph.keys()))[self.ORDER])
        print(default_order)

        return [
            *default_order[default_order.index(key):],
            *default_order[:default_order.index(key)]
        ]
        

    def walk_imitation(self, walk_route):
        investigated = []
        for i in walk_route:
            edges = self.graph[i]
            for ed in [i, *edges]:
                if ed in investigated:
                    continue            
                investigated.append(ed)
        print(investigated)
        return
    

        

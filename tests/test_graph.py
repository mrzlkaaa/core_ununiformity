from . import *
from main.graph import Graphs

@pytest.fixture
def g():
    # nodes = [
    #     "7-6",	"7-5",	"7-4",	"7-3",	
    #     "6-6",	"6-5",	"6-4",	"6-3",	
    #     "5-6",	"5-3",	"4-6",	"4-3",
    #     "3-6",	"3-5",	"3-4",	"3-3",
    #     "2-6",	"2-5",	"2-4",	"2-3"
    # ]
    return Graphs(prefix="_s")

def test_find_pairs(g):
    res = g._find_pairs("7-6_s")
    print(res)
    assert 0

def test_make_graph(g):
    g._make_graph()
    print(g.graph)
    print(g.graph.keys())

    assert 0

def test_make_walk_route(g):
    g._make_graph()
    walk = g.make_walk_route("5-3")
    print(walk)
    assert 0

def test_walk_imitation(g):
    g._make_graph()
    print(g.graph)
    walk = g.make_walk_route("5-3")
    g.walk_imitation(walk)
    assert 0
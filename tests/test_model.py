from audioop import reverse
from re import M
import pytest
import numpy as np


from main.model import GA, Individual

fuel_map = [
    300,
    300,
    300,
    300,
    265,
    265,
    265,
    265,
    265,
    300,
    300,
    300,
    265,
    265,
    265,
    265,
    265,
    300,
    300,
    300
]

fuel_map_mask = [
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1
]

core = [
    49.02,
    57.33,
    14.28,
    39.40,
    38.23,
    37.67,
    11.55,
    22.21,
    51.15,
    32.79,
    40.88,
    0.0,
    35.61,
    10.12,
    39.14,
    20.31,
    37.64,
    16.42,
    56.55,
    42.8
]

chromo1 = {
    'id': 12, 
    'fuels_gnome': np.asarray([
        57.33, 42.8 , 16.42, 49.02, 38.23, 37.67, 11.55, 22.21, 51.15,
        56.55, 14.28, 40.88, 35.61, 10.12, 39.14, 20.31, 37.64, 32.79,
        0.0  , 39.41
    ]), 
    'core_burnup': 32.7, 
    'permutations_made': 10, 
    'p_margin': 6.530588087166127, 
    'k_fa_max': 1.2997157702212703, 
    'k_fa_min': 0.5918006153185372, 
    'k_left_right': 0.8489624433926495, 
    'fitness_score': 0.8237099967666309
}

chromo2 = {
    'id': 4, 
    'fuels_gnome': 
        [
            14.28, 42.8 , 56.55, 49.02, 38.23, 37.67, 11.55, 22.21, 51.15,
            0.0  , 16.42, 40.88, 35.61, 10.12, 39.14, 20.31, 37.64, 57.33,
            39.41, 32.79
    ], 
    'core_burnup': 32.7, 
    'permutations_made': 10, 
    'p_margin': 5.160896165530887, 
    'k_fa_max': 1.6475000288936654, 
    'k_fa_min': 0.6454148784654546, 
    'k_left_right': 0.8263881059863879, 
    'fitness_score': 0.7050955136207331
}


fresh_fuel_test =  {
    'id': 0,
    'fuels_gnome': np.array([
        63.65, 13.93, 3.02, 46.18, 
        61.33, 22.97, 48.86, 55.61, 
        26.14, 33.11, 
        59.51, 33.05, 
        59.02, 44.45, 22.89, 52.94, 
        39.72, 5.78, 13.69, 45.26
    ]),
    'core_burnup': 37.6,
    'p_margin': 4.78612546174803,
    'k_fa_max': 1.3474877200592554,
    'k_fa_min': 0.619508928948087,
    'k_left_right': 0.8231629459866739,
    'k_sym': 0.8029373242713104,
    'fitness_score': 0.7319066213673546
}



@pytest.fixture
def ga():
    return GA.no_fuel_mask(
        core=fresh_fuel_test['fuels_gnome'],
        fuel_map=fuel_map,
        population_size=40,
        refuel_only=[0, 10] #* refuel only provided
    )

def test_find_fuel_gnome(ga):
    indexes, fuels = ga.dynamic_fuels_gnome_ind, ga.dynamic_fuels_gnome
    print(indexes, fuels)
    assert 0

def test_initilize_fresh_fuel_probability_exp(ga):
    res = ga._initilize_fresh_fuel_probability_exp(30.0)
    print(res)
    res = ga._initilize_fresh_fuel_probability_exp(40.0)
    print(res)
    assert 0

def test_make_population(ga):
    res = ga.make_population()
    print(res)
    assert 0

def test_permutation_mutation(ga):
    res = ga.permutation_mutation(chromo1)
    print(res)
    assert 0

def test_fresh_fuel_mutation(ga):
    res = ga.fresh_fuel_mutation(fresh_fuel_test)
    #* imitation of mutation of core
    #* but class stores cell value that must be refueled
    res["fuels_gnome"][10] = 3.02
    res["fuels_gnome"][2] = 59.51
    res = ga.fresh_fuel_mutation(res)
    print(res)
    assert 0

def test_mate(ga):
    res = ga.mate(
        chromo1,
        chromo2
    )
    print(res)
    assert float("{:.1f}".format(res.mean())) == chromo1["core_burnup"]

def test_replace_chromosome(ga):
    population = ga.make_population()
    ga._replace_chromosomes(population, [chromo1, chromo2])
    assert 0

def test_search(ga):
    res = ga.search()
    assert 0

family = [
    {
        'id': 0,
        'fuels_gnome': [40.88, 57.33, 16.42, 56.55, 38.23, 37.67, 11.55, 22.21, 51.15,
                32.79, 42.8, 49.02, 35.61, 10.12, 39.14, 20.31, 37.64, 14.28,
                39.41, 0.0],
        'core_burnup': 32.7,
        'permutations_made': 1,
        'p_margin': 6.264710171243173,
        'k_fa_max': 1.2799523948592075,
        'k_fa_min': 0.6421221713926826,
        'k_left_right': 0.7906317290459922,
        'fitness_score': 0.7930891685359678
    },
    {
        'id': 11,
        'fuels_gnome': [40.88, 57.33, 16.42, 49.02, 38.23, 37.67, 11.55, 22.21, 51.15,
                56.55, 42.8, 39.41, 35.61, 10.12, 39.14, 20.31, 37.64, 14.28,
                32.79, 0.0],
        'core_burnup': 32.7,
        'permutations_made': 9,
        'p_margin': 6.249076973368739,
        'k_fa_max': 1.2960795467073933,
        'k_fa_min': 0.6284156463974351,
        'k_left_right': 0.7999601929958408,
        'fitness_score': 0.7930448873088508
    },
    {
        'id': 30,
        'fuels_gnome': [40.88, 0.0, 49.02, 14.28, 38.23, 37.67, 11.55, 22.21, 51.15, 32.79,
                56.55, 42.8, 35.61, 10.12, 39.14, 20.31, 37.64, 16.42, 39.41,
                57.33],
        'core_burnup': 32.7,
        'permutations_made': 8,
        'p_margin': 6.219735697108348,
        'k_fa_max': 1.430432171934923,
        'k_fa_min': 0.6638351843144914,
        'k_left_right': 0.8421029577700472,
        'fitness_score': 0.7874860697425516
    }
]

# def _test_tournament_selection(ga):
#     ga


fuels_gnome_ind = [0, 1, 2, 3, 9, 10, 11, 17, 18, 19]
fuels_gnome = [
    49.02, 57.33, 
    14.28, 39.40, 
    32.79, 40.88, 
    0.0, 16.42, 
    56.55, 42.8
]

@pytest.fixture
def indiv():
    return Individual(
        fuel_map=fuel_map_mask,
        dynamic_fuels_gnome_ind=fuels_gnome_ind,
    )

def test_fuels_gnome_mutation(indiv):
    mutated = indiv.fuels_gnome_mutation(core, 3)
    print(mutated)
    assert 0

def test_fuels_gnome_refactorer(indiv):
    df = indiv._fuels_gnome_refactorer(indiv, fuels_gnome=core)     
    print(df)
    assert 0

def test_initialize_chromosome(indiv):
    mutated = indiv.fuels_gnome_mutation(core)
    res = indiv.initialize_chromosome(mutated, 0)
    print(res)
    assert 0
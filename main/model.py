

from re import search
from main import burnup
from typing import Iterable
from joblib import load
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

import numpy as np
import pandas as pd
import random
import math
import concurrent.futures
import time

from regression.stacking import Stacking


try:
    from . import *
except ImportError:
    from main import *

core_parts = init_core_parts()
lock = Lock()

#! dependency afctually
#* passing models as class variables
pmargin = load("pmargin_v1.0.joblib")
nonuniformity = load("new_stack_unifor_v1.5.joblib")
#* column transformere required
# nonuniformity_no_ensemble = load("GPR_model.joblib")

# FITNESS_WEIGHTS_WINDOW = [
#     0.40,    
#     0.15,    
#     0.10,    
#     0.35     
# ]

FITNESS_WEIGHTS_WINDOW = {
    "p_margin": 0.25,    #* margin
    "k_fa_max": 0.25,    #* k_fa_max
    "k_quarter": 0.25,   #* k_left_right
    "k_sym": 0.25        #* k_sym
}

TARGET = [
    9,
    1.4,
    1,
    1
] 

class Individual:

    #* The aim of the class is
    #* to handle the chromosome.
    #* The list of operations under the class
    #*  - build / initialize chromosome
    #*  - score the chromosome (get the quality)
    #*  - matting of chromosomes
    #*  - random non uniform mutation (permutations)
    #*  - random non uniform  mutatuion (fresh fuel)
    #* 
    #* At the momemt chromosome consists of following data:
    #*  - id - identificator of chromosome
    #*  - fuel_gnome - core burnup cartogram
    #*  - core_burnup - average burnup of a core
    #*  - p margin - reactor core margin
    #*  - k fa max - most stressfull fuel assembly
    #*  - k fa min - less stressfull fuel assembly
    #*  - k left_right - relation of left to right non uniformity coefs
    #*      that desctibes the skeweness      
    #*  - fitness score - the score (quality) of chromosome 
    #*
    
    # ENERGY_RELEASED_MODEL
    # P_MARGIN_MODEL
    # FUEL_MAP = 
    def __init__(
        self,
        fuel_map,
        dynamic_fuels_gnome_ind,
        fitness_weights: dict | None = None
    ):
        self.fuel_map = fuel_map
        #* it's array of indexes where 8th tube FA is installed
        self._dynamic_fuels_gnome_ind = np.asarray(dynamic_fuels_gnome_ind)

        self.fintess_weights = fitness_weights if fitness_weights \
            else FITNESS_WEIGHTS_WINDOW
        

    @property
    def dynamic_fuels_gnome_ind(self):
        return self._dynamic_fuels_gnome_ind

    @dynamic_fuels_gnome_ind.setter
    def dynamic_fuels_gnome_ind(self, val: list):
        self._dynamic_fuels_gnome_ind = val

    @staticmethod
    def round_float(val):
        return float("{:.1f}".format(val))

    @staticmethod
    def _fuels_gnome_refactorer(
        cls: object,
        fuels_gnome: np.ndarray
    ):
        '''
        #* Makes DataFrame from an array
        #* Refactorization required to use non-inofromity
        #* regression model that accepts labeled data
        #* The input X for a model is
        #* Data frame with _l and _b columns
        #* shape of Data frame is (, 40)
        #* Parameters
        #* ----------
        #* fuels_gnome: nd.ndarray
        #*  array of burnup data
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #* DataFrame
        '''
        burnup_cols = core_parts["burnup"]["ALL_CELLS"]
        left_cols = core_parts["left"]["ALL_CELLS"]

        df = pd.DataFrame(
            columns=[
                *burnup_cols,
                *left_cols

            ]
        )
        burnup_data = []
        left_data = []
        #* fg is burnup in units of %
        for n,fg in enumerate(fuels_gnome):
            if cls.fuel_map[n]:
                burnt = fg / 100 * 300
                burnup_data.append(burnt)
                left_data.append(300 - burnt)
                continue

            burnt = fg / 100 * 265
            burnup_data.append(burnt)
            left_data.append(265 - burnt)

        df.loc[0, :] = [
            *burnup_data, 
            *left_data
        ]        

        return df

    #! suppressed
    def count_matched_cells(
        self,
        cells1,
        cells2
    ):
        counter = 0
        for n in range(len(cells1)):
            if cells1[n] == cells2[n]:
                counter +=1
        
        return counter

    def fuels_gnome_mutation(
        self, 
        ancestor: list | np.ndarray,
        mutation_points: int | None = None
    ):
        '''
        #* Generates random fuels_gnome (core burnup map)
        #* for a chromosome. 
        #* random.sample is used to generate pseudo random array
        #* from a given ancestor's fuels gnome (initial gnome).
        #* Method is capable to generate fuels_gnome array with a given
        #* number of mutations. If mutation_points is None so whole ancestor's
        #* gnome resampled. Resampling do not emit repetions'
        #* If mutation_points is given method randomly selects gnomes that
        #* will be resampled. Stores the indexes of selected sampeles
        #* Only selected gnomes replaces in fuels_gnome (for now it's 8th tube only)
        #* So any number of permutation can be made of a given core
        #* Parameters
        #* ----------
        #*  ancestor: list | np.ndarray
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #* Generated pseudo-random array of fuels_gnome
        '''
        if isinstance(ancestor, list):
            ancestor = np.asarray(ancestor)

        mutated = ancestor.copy()

        if mutation_points is None:
            mutation_points = len(self.dynamic_fuels_gnome_ind)
            indexes = self.dynamic_fuels_gnome_ind
        else:
            indexes = random.sample(list(self.dynamic_fuels_gnome_ind), k=mutation_points)
        
        gnomes_to_mutate = ancestor[indexes].tolist()
        sampled_gnomes = np.asarray(random.sample(gnomes_to_mutate, k=mutation_points))
        
        mutated[indexes] = sampled_gnomes

        return mutated
    
    def initialize_chromosome(
        self,
        fuels_gnome: np.ndarray,
        id_num: int = 0,
        **kwargs
    ):
        '''
        #* Main method of class to initialize
        #* chromosome gnomes
        #* Method accepts the < fuels_gnome > array as a basis
        #* argument to initialize gnomes
        #* One of the most important part is
        #* generation of non-uniformity coefficients for a
        #* given array. Based on generated coefsefficients
        #* Following gnomes initializes:
        #*  k_fa_max - most stressful fuel assembly
        #*  k_fa_min - less stressful fuel assembly
        #*  k_quarters - by qurter averaged non-uniformity coef
        #*      in a range from 0 to  1
        #*  k_sym - coef of symmetry between pairwised cells
        #*      that shows how differ / close pairwised cells to each other
        #*      in a range from 0 to 1
        #*  fitness_score - score of chromosome to evaluate it during population
        #*      assessment step
        #* Parameters
        #* ----------
        #*  fuels_gnome: np.ndarray
        #*      fuel burnup map (7-6 to 2-3 cells)
        #*  id_num: int
        #*      identificator number of a chromosome
        #*      uses to provide unique number / gnome to a chromosome
        #*      This number helps to modify / replace existing chromosome 
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        #* if mutation is set to False
        #* Generates chromosome data for an ancestor
        #* structure of chromosome
        #* [
        #*      p, k_fa_max, k_fa_min, k_left, k_right, avb,
        #*      fuel_burnup_map  
        #* ]

        core_burnup = self.round_float(fuels_gnome.mean()) 
        #* parameters of fuels_gnome
        refactored_fuels_gnome = self._fuels_gnome_refactorer(self, fuels_gnome)
        fuels_non_uniformity = Stacking.predict(refactored_fuels_gnome, nonuniformity).to_numpy()[0]
        k_fa_max, k_fa_min = fuels_non_uniformity.max(), fuels_non_uniformity.min()
        #! fix this as it's done for k_sym
        k_left_right = fuels_non_uniformity[core_parts["index"]["LEFT_CENTER_SIDE"]].mean() / fuels_non_uniformity[core_parts["index"]["RIGHT_CENTER_SIDE"]].mean()
        
        k_quarters = [
            fuels_non_uniformity[core_parts["index"]["QUL"]].mean(), 
            fuels_non_uniformity[core_parts["index"]["QUR"]].mean(),
            fuels_non_uniformity[core_parts["index"]["QLL"]].mean(),
            fuels_non_uniformity[core_parts["index"]["QLR"]].mean(),
        ]

        #* the 4 sector core (quarters) is used to get non-uniformity
        #* between each part
        k_quarter = [
            k_quarters[1] / k_quarters[0],   #* QUR / QUL
            k_quarters[3] / k_quarters[2],   #* QLR / QLL
            k_quarters[1] / k_quarters[3],   #* QUR / QLR
            k_quarters[0] / k_quarters[2],   #* QUL / QLL
            k_quarters[1] / k_quarters[2],   #* QUR / QLL
            k_quarters[0] / k_quarters[3]    #* QUL / QLR
        ]

        k_quarter = np.mean(list(map(lambda x: x if x <= 1 else 1 - ( x - 1 ), k_quarter)))

        sym_coefs = [
            fuels_non_uniformity[0] / fuels_non_uniformity[16],  #* 7-6 and 2-6
            fuels_non_uniformity[1] / fuels_non_uniformity[18],  #* 7-5 and 2-4
            fuels_non_uniformity[2] / fuels_non_uniformity[17],  #* 7-4 and 2-5
            fuels_non_uniformity[3] / fuels_non_uniformity[19],  #* 7-3 and 2-3
            fuels_non_uniformity[8] / fuels_non_uniformity[10],  #* 5-6 and 4-6
            fuels_non_uniformity[9] / fuels_non_uniformity[11],  #* 5-3 and 4-3
            
        ]
        sym_coefs = np.mean(list(map(lambda x: x if x <= 1 else 1 - ( x - 1 ), sym_coefs)))

        sym_burnup = [
            fuels_gnome[0] - fuels_gnome[16],  #* 7-6 and 2-6 
            fuels_gnome[1] - fuels_gnome[18],  #* 7-5 and 2-4
            fuels_gnome[2] - fuels_gnome[17],  #* 7-4 and 2-5
            fuels_gnome[3] - fuels_gnome[19],  #* 7-3 and 2-3
            fuels_gnome[8] - fuels_gnome[10],  #* 5-6 and 4-6
            fuels_gnome[9] - fuels_gnome[11]   #* 5-3 and 4-3
        ]
        sym_burnup = np.mean(list(map(lambda x: np.power(1.05, 0) / np.power(1.05, abs(x)), sym_burnup)))
        
        k_sym = np.array([sym_coefs, sym_burnup]).mean()
            # fuels_non_uniformity[core_parts["index"]["USYMM"]].mean() / fuels_non_uniformity[core_parts["index"]["LSYMM"]].mean()
        
        refactored_fuels_gnome.loc[:, "average_l"] = refactored_fuels_gnome.loc[:, core_parts["left"]["ALL_CELLS"]].mean(axis=1)
        refactored_fuels_gnome.loc[:, "average_b"] = refactored_fuels_gnome.loc[:, core_parts["burnup"]["ALL_CELLS"]].mean(axis=1)
        #* predicting of p_margin
        p_margin = pmargin.predict(
            refactored_fuels_gnome.loc[
                :, 
                [
                    *core_parts["left"]["ALL_CELLS"], 
                    "average_l",
                    "average_b"
                ]
            ]
        )[0]
        
        #! symmetry in terms of fuel burnup to help find pair more correct 

        (
            fitness_score,
            p_margin,
            k_fa_max,
            k_quarter,
            k_sym

        ) = self.fitness_score(
            self,
            p_margin,
            k_fa_max,
            k_quarter,
            k_sym
        )

        #* now i need to call some model to get 
        #* distribution of energy released
        #* and model to get p margin for a given core (fuels_gnome)

        return {
            "id": id_num,
            "fuels_gnome": fuels_gnome.copy(),
            "core_burnup": core_burnup,
            "p_margin": p_margin,
            "k_fa_max": k_fa_max,
            "k_fa_min": k_fa_min,
            "k_quarter": k_quarter,
            "k_sym" : k_sym,
            "k_left_right": k_left_right,
            "fitness_score": fitness_score
        }

    @staticmethod
    def fitness_score(
        cls,
        p_margin: float,
        k_fa_max: float,
        k_quarter: float,
        k_sym: float

    ) -> float:
        '''
        #* Computates fitness score for a given set of parameters
        #* Fitness function is sum of normalized and weighted variables
        #* It calls to provide fitness score for chromosome
        #* The max normalization does here to scale values to range (0;1)
        #* Max values are taken from TARGET
        #* Normalized values multiplies by weights in order to buff some 
        #* of values
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        #* max normalization
        if p_margin > TARGET[0]:
            #* show exponential decrease of coeff if p_margin > P-margin_limit
            p_margin_norm = TARGET[0]/p_margin*np.exp(-5 * (p_margin - TARGET[0]))
        else:
            p_margin_norm = p_margin / TARGET[0]
        
        # #* k_fa_max normalizations
        # if k_fa_max < 1.2:
        #     k_fa_max = 1.2

        if k_fa_max < TARGET[1]:
            k_fa_max_norm = 1 / ( k_fa_max / TARGET[1] )
        
        else:
            #* 1 / ( k_fa_max / TARGET[1] ) is scale factor because
            #*  exp result in a range of [ 0, 1 ]
            #* exp is used to make k_fa_max coef to decrease rapidly
            k_fa_max_norm = 1 / ( k_fa_max / TARGET[1] ) * np.exp(-5 * ( k_fa_max - TARGET[1]) )
        

        #* k_quarter processing
        k_quarter_norm = k_quarter / TARGET[2] # if k_quarter <= 1 else ( 1 / k_quarter ) / TARGET[2] 
        #* normalize to exponential decrease in a scale of [ 0, 1 ]
        k_quarter_norm = np.exp( ( k_quarter_norm - 1) )
        #* symm processing
        k_sym_norm = k_sym / TARGET[3] # if k_sym <= 1 else ( 1 / k_sym ) / TARGET[3]
        #* normalize to exponential decrease in a scale of [ 0, 1 ]
        k_sym_norm = np.exp( ( k_sym_norm - 1) )
        # k_sym_norm = k_sym / LIMITS[3] if k_sym <= 1 else ( 1 / k_sym ) / LIMITS[3]
        
        return (
            p_margin_norm * cls.fintess_weights["p_margin"]\
            + k_fa_max_norm * cls.fintess_weights["k_fa_max"]\
            + k_quarter_norm * cls.fintess_weights["k_quarter"]\
            + k_sym_norm * cls.fintess_weights["k_sym"],
            p_margin_norm,
            k_fa_max_norm,
            k_quarter_norm,
            k_sym_norm
        )

#! provide max_worker argument
#* this will speed up the prediction models work
#* and iteration loops 
class GA:
    MIN_BURNUP = 28.0
    MAX_BURNUP = 40.0
    #* selection
    #* evolution 
    def __init__(
        self,
        core: list,  #* it's array size of 20
        fuel_map: list,  #* map shows where 8th/6th tube FA are installed
        population_size: int,
        refuel_only: list | None = None,  #* restrict cells avaliable for refueling
        full_symmetry: bool = True,  #* allows refueling for pairwised cells only
        mutation_probabilty_fresh_fuel: float = 0.1,
        permutation_mutation_probability:float = 0.1,
        mate_probability: float = 0.2,
        elitism:float = 0.1,
        fintess_weights: dict = FITNESS_WEIGHTS_WINDOW,
        workers: int = 1
        
        # population: Iterable

    ):
        self.ancestor_core = core
        self.fuel_map = fuel_map
        self.dynamic_fuels_gnome_ind, self.dynamic_fuels_gnome = self._find_fuel_gnome(core, fuel_map)
        self.workers = workers

        self.indiv = Individual(
            fuel_map,
            self.dynamic_fuels_gnome_ind,
            fintess_weights
        )
        #* find and store values in a given cells
        #* to refuel exactly given one
        self.refuel_only = np.asarray(self.ancestor_core)[np.asarray(refuel_only)] if refuel_only is not None else None
        self.full_symmetry = full_symmetry
        
        self.population_size = population_size
        self.mutation_probabilty_fresh_fuel = mutation_probabilty_fresh_fuel
        self.permutation_mutation_probability = permutation_mutation_probability
        self.mate_probability = mate_probability
        self.elitism = elitism
        

        self._best_per_iter = []
        self._aver_score_per_iter = []
    
    @classmethod
    def no_fuel_mask(
        cls,
        core,
        fuel_map,
        population_size,
        refuel_only: list | None = None,
        full_symmetry: bool = True,
        mutation_probabilty_fresh_fuel: float = 0.1,
        permutation_mutation_probability:float = 0.1,
        mate_probability: float = 0.2,
        elitism:float = 0.1,
        fintess_weights: dict = FITNESS_WEIGHTS_WINDOW,
        workers:int = 1
    ):
        print(fintess_weights)
        fuel_map = [
            1 if i == 300 else 0 for i in fuel_map
        ]
        return cls(
            core=core, 
            fuel_map=fuel_map,
            population_size=population_size,
            refuel_only=refuel_only,
            full_symmetry=full_symmetry,
            mutation_probabilty_fresh_fuel=mutation_probabilty_fresh_fuel,
            permutation_mutation_probability=permutation_mutation_probability,
            mate_probability=mate_probability,
            elitism=elitism,
            fintess_weights=fintess_weights,
            workers=workers
        )

    def _find_fuel_gnome(
        self,
        core,
        fuel_map
    ):
        indexes = []
        fuels = []

        for n,i in enumerate(fuel_map):
            if i:
                indexes.append(n)
                fuels.append(core[n])

        return indexes, fuels

    def _min_max_norm(self, val, min, max):
        return (val - min) / (max - min)

    @staticmethod
    def _fitness_score(
        chromosome: dict,
    ):
        return chromosome["fitness_score"]

    def _initilize_fresh_fuel_probability_log(
        self, 
        population_burnup: float,
        scale = 0.1
    ):
        if population_burnup < 28.0:
            return 0.0

        population_burnup_norm = population_burnup / self.MAX_BURNUP
        min_burnup_norm = self.MIN_BURNUP / self.MAX_BURNUP
        max_burnup_norm = self.MAX_BURNUP / self.MAX_BURNUP

        return self._min_max_norm(
            np.log(population_burnup_norm), 
            np.log(min_burnup_norm), 
            np.log(max_burnup_norm)
        ) * scale


    def _get_population_average(
        self,
        population: list,
        key:str
    ):
        return np.asarray(list(map(lambda x: x[key], population))).mean()

    def _initilize_fresh_fuel_probability_exp(
        self, 
        population_burnup: float,
    ):
        if population_burnup < 28.0:
            return 0.0

        population_burnup_norm = self._min_max_norm(
            population_burnup, 
            self.MIN_BURNUP, 
            self.MAX_BURNUP
        )
        #* return exonential probability normalizaed by max value
        return np.exp(2.0 * population_burnup_norm) / np.exp(2.0) * self.mutation_probabilty_fresh_fuel

    def _initilize_mutation_probability(
        self, 
        generation: int, 
    ):
        return ( (1/np.power(1.5, generation)) / (1/np.power(1.5, 1)) ) * self.permutation_mutation_probability

    
    def fresh_fuel_mutation(
        self,
        chromosome:dict,
        
    ):
        #* do nothing and return initial chromosome
        if chromosome["core_burnup"] < 30:
            return chromosome

        #* overall idea is to modify fuels_gnome 
        #* by replacing most burnup gnome
        #*  - find most burnup fuel gnome
        #*  - replace
        #*  - initialize new chromosome and return it
        fuels_gnome = chromosome["fuels_gnome"].copy()

        #* enable user preference refueling pattern
        if self.refuel_only is None:
            #* all cells are avaliable
            cells = np.asarray(core_parts["index"]["ALL_CELLS"])
        else:
            #* find positions of to_refuel gnomes (FAs)
            #* do search the values cuz position may be changed
            cells = []
            for fa in self.refuel_only:
                try:
                    cells.append(
                        list(fuels_gnome).index(fa)
                    )    
                except ValueError:
                    #* raises when some / all elements to refuel
                    #* have been refueled for a given chromosome
                    continue
        
        #* if all cells (gnomes) have been refueled
        if not len(cells) > 0:
            return chromosome

        
        #* to enable 6th tube refueling 
        #* pairwise refueling required
        max_burnup = fuels_gnome[cells].max()
        max_burnup_pos = list(fuels_gnome).index(max_burnup)

        #* full_symmetry means that all cells in a core have pair
        #! BUT symmetric of FAs can be broken due to this method
        #! accepts metated cores -> refueling of 8th tube FA by index is forbidden 
        #* Refeuling of symmetric 8th tubes MUST BE implemeted by values only
        #* if full_symmetry is False only 6th tube FA are considered as pairwised
        if self.full_symmetry:
            pairwised_cells = [
                *core_parts["index"]["PAIRWISED_RODS"]["CELLS"], 
                *core_parts["index"]["PAIRWISED_SYMM"]["CELLS"]
            ]
            pairwised_pairs = [
                *core_parts["index"]["PAIRWISED_RODS"]["PAIRS"], 
                *core_parts["index"]["PAIRWISED_SYMM"]["PAIRS"]
            ]
        else:
            pairwised_cells = core_parts["index"]["PAIRWISED_RODS"]["CELLS"]
            pairwised_pairs = core_parts["index"]["PAIRWISED_RODS"]["PAIRS"]
        
        #* need to check whether cell is pairwised
        #* if cell in a array of pairwised cells
        #* loop over pair to get cell to refuel
        if max_burnup_pos in pairwised_cells and not self.full_symmetry:
            for v in pairwised_pairs:
                if max_burnup_pos in v:
                    #* repcale both FAs by fresh ones based on indexes
                    fuels_gnome[np.array(v)] = 0.0
                    break
        
        #* finds true pair of max_burnup FA by checking ancestor_core
        #* applicable for both 6th and 8th tube FAs
        elif self.full_symmetry:
            #* finds max_burnup FA in ancestor's core
            max_burnup_pos = list(self.ancestor_core).index(max_burnup)
            max_burnup_pair_pos_ancestor = []

            #* search for an unchanged pair of max_burnup
            #* stores indexes of pairs
            for pair in pairwised_pairs:
                if max_burnup_pos in pair:
                    max_burnup_pair_pos_ancestor = pair
                    break
            #* retrieving of values of pair
            max_burnup_pair = self.ancestor_core[max_burnup_pair_pos_ancestor]

            #* do search for rettieved values in fuel_gnome 
            #* and store new positions
            max_burnup_pair_pos = []
            for val in max_burnup_pair:
                
                true_pos = list(fuels_gnome).index(val)
                
                max_burnup_pair_pos.append(
                    true_pos
                )

            #* refueling of a pair of FAs
            fuels_gnome[max_burnup_pair_pos] = 0.0

        #* refuel not pairwised cell
        else:
            fuels_gnome[max_burnup_pos] = 0.0
        
        #* ! future look up !
        #* to prevent new cores creation with average burnup < 28
        #* The first condition checks burnup of a given
        #* chromosome. But what if the core
        #* passes condinion and fresh fuel installed
        #* and at the end we see that average burnup is under 28%
        #* So it is actually breaks the logic
        #* ! I do propose early mentioned future lookup !
        #*  where we do install fresh fuel but before calling
        #* chromosome initialization check new core burnup
        #* IN CASE it's too fresh we just return given chromosome
        #* IF new core burnup is above 28% it is OK
        if fuels_gnome.mean() < 28.0:
            return chromosome

        new_chromosome = self.indiv.initialize_chromosome(
            fuels_gnome=fuels_gnome,
            id_num=chromosome["id"]
        )

        return new_chromosome

    def permutation_mutation(
        self,
        chromosome: dict,
    ):
        mutated_fuels_gnome = self.indiv.fuels_gnome_mutation(
            ancestor=chromosome["fuels_gnome"],
            mutation_points=6
        )

        #* some recursion to prevent returning of duplicate from mutation
        if list(mutated_fuels_gnome) == list(chromosome["fuels_gnome"]):
            return self.permutation_mutation(chromosome)

        mutated_chromosome =  self.indiv.initialize_chromosome(
            mutated_fuels_gnome,
            chromosome["id"]
        )
        return mutated_chromosome

    def mate_tournament_selection(
        self,
        chromosomes,
        out: int = 2,  #* size of output array
        
    ):  
        #! implemented only for a single crossover
        #! size of input array must be 3
        '''
        #* Performs tournament selection of chromosomes
        #* Ones who has lowest fitness score drops from
        #* array
        #* Selection performs on a given array of chromosomes
        #* The number of chromosomes to drop during selection
        #* computates as len(chromosomes) - out (output size of array)
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        #* sort by descending of fitness score
        chromosomes.sort(reverse=True, key=self._fitness_score)
        #* gettting position of offspring
        offspring_pos = list(map(lambda x: x["id"], chromosomes)).index(0)

        if offspring_pos < 2:
            #* offspring is stronger than some of parents
            #* so he is going to replace one of them
            #* and replaces the weakest one
            chromosomes[offspring_pos]["id"] = chromosomes[2]["id"]
            chromosomes[2]["id"] = 0

        #* pop weakest chromosome
        chromosomes.pop()

        return chromosomes

    def _block_fuel_gnome(
        self,
        val,
        offspring,
        occupied_gnomes,
        p1,  #* parent whose gnome was not taken for offspring
        p2   #* parent who was choosen to take a gnome
    ):
        '''
        #* Special method that used during matting
        #* It's required to prevent offspring core from
        #* duplicates / broken logic
        #* Why it is so IMPORTANT
        #* Usually, matting means random / disordered gnomes exchange
        #* so some genes can be repeated
        #* But for reactor core case it's imposiible
        #* Here we do have finite number of cells (gnomes)
        #* And if we do take gnome from parent 1 we need to block
        #* gnome (index of this gnome) of parent 2 with same value we just got from parent 1
        #* blocked index of parent 2 cannot be used for offspring anymore
        #* so offspring populates by gnome from parent 1
        #* 
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        for i in range(len(p1)):
        # search for direct block
            if val == p1[i] and offspring[i] is None and p2[i] not in occupied_gnomes:
                # print(f"blocks the {i} cell with value {p1[i]} in second parent")
                # print(f"blocked cell {i} is replaced by {p2[i]}")
                
                offspring[i] = p2[i]
                occupied_gnomes.append(p2[i])

        return offspring, occupied_gnomes
    
    def mate(
        self,
        chromosome1,
        chromosome2
    ):
        '''
        #* Method description
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        p1 = chromosome1["fuels_gnome"]
        p2 = chromosome2["fuels_gnome"]

        if not chromosome1["core_burnup"] == chromosome2["core_burnup"]:
            # print(f"cannot mate core with different burnup: {chromosome1} != {chromosome2}")
            return None

        offspring = np.full((20,), None)
        occupied_gnomes = []

        for i in range(len(p1)):

            turn = random.randint(0,1)
            if p1[i] == p2[i]:
                offspring[i] = p1[i]
                occupied_gnomes.append(p1[i])
            elif turn % 2 == 0:
                # print(i, f" with value of {p1[i]} from 1st parent is candidate for offspring")
                if offspring[i] is None and p1[i] not in occupied_gnomes:
                    # print(f"offspring gnome {i} populated by {p1[i]}")
                    offspring[i] = p1[i]
                    occupied_gnomes.append(p1[i])

                    offspring, occupied_gnomes = self._block_fuel_gnome(p1[i], offspring, occupied_gnomes, p2, p1)
                
            else:
                # print(i, f"with value of {p2[i]} from 2nd parent is candidate for offspring")
                if offspring[i] is None and p2[i] not in occupied_gnomes:
                    # print(f"offspring gnome {i} populated by {p2[i]}")
                    offspring[i] = p2[i]
                    occupied_gnomes.append(p2[i])
        
                    offspring, occupied_gnomes = self._block_fuel_gnome(p2[i], offspring, occupied_gnomes, p1, p2)
        
        
        #* check whether any cell is missing
        #* if missing values found
        #* populate offspring array with None values
        #* by value that were found
        # try:
        ind = np.where(offspring == None)[0]
        # print(offspring, ind)
        match_missing =set()
        for p in [p1, p2]:
            [match_missing.add(i) for i in set(p).difference(set(occupied_gnomes))]
        
        # print(set(p1).difference(set(p2)))
        # print(ind, match_missing)
        for indx, val in zip(ind, match_missing):
            offspring[indx] = val
        # except Exception as e:
        #     print(e)
        # print(offspring)
        #* if at this moment there are cells with None
        #* it means that is stucked on multiple 0.0 cells
        #! temporary fix is to set 0.01 at fresh fuel mustation
        ind = np.where(offspring == None)[0]
        if len(ind) > 0:
            for i in ind:
                offspring[i] = 0.0

        offspring_chromosome = self.indiv.initialize_chromosome(
            offspring,
            id_num=0,
        )

        return offspring_chromosome


    def _replace_chromosomes(
        self,
        population: list,
        chromosomes: list | dict
    ):
        '''
        #* Internal method to replace chromosomes
        #* in population
        #* It is used to keep population size same 
        #* during search (evolution) process
        #* That is why bad chromosome replaces by good one
        #* but ID of chromosome keeps untouched
        #* Population and candidates (chromosomes) passes to a method
        #* Then searches for candidate's ID in population and replace
        #* these chromosomes by candidates (chromosomes)
        #* Parameters
        #* ----------
        #*  population: list
        #*      population of chromosomes where to seek
        #*      for candidate's (chromosomes) ID
        #*  chromosomes: list | dict
        #*      candidates to replace bad chromosomes in a given population
        #*      Replacement based on ID's search
        #* Raises
        #* ----------
        #*  None
        #* Returns
        #* ----------
        #*  population: list
        #*      population with replaced chromosomes (updated population)
        '''
        #* copy to prevent inplace replacement
        population = population.copy()
        
        #* if the only one chromosome is passed to func
        if not isinstance(chromosomes, list):
            chromosomes = [chromosomes]

        #* getting ids of given chromosomes
        ids = list(map(lambda x: x["id"], chromosomes))
        #* getting ids of population
        pop_ids = list(map(lambda x: x["id"], population))
        # print(ids, pop_ids)
        #* getting positions of ids from given chromosomes
        ids_pos = [pop_ids.index(i) for i in ids]
        
        for pos, chromo in zip(ids_pos, chromosomes):
            # print(population[pos], chromo)
            population[pos] = chromo

        return population 

        

    def make_population(self):
        '''
        #* Makes initial population of chromosomes
        #* Ansestor core is a basis to create all initial
        #* chromosomes. Chromosomes create via random self.fuels_gnome_mutation
        #* List sorts by fitness score to be used correctly in search process
        #* Parameters
        #* ----------
        #*  None
        #* Raises
        #* ----------
        #*  None
        #* Returns
        #* ----------
        #* population: list
        '''
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            #todo can be speeded up
            population = []
            mutated_cores = [self.indiv.fuels_gnome_mutation(self.ancestor_core) for _ in range(self.population_size)]
            # print(mutated_cores)
            futures = [
                executor.submit(self.indiv.initialize_chromosome, mutated_fuels_gnome, i+1) for i, mutated_fuels_gnome in enumerate(mutated_cores)
            ]
            for future in as_completed(futures):
                
                population.append(
                    future.result()
                )
            # for i in range(self.population_size):
            #     mutated_fuels_gnome = self.indiv.fuels_gnome_mutation(self.ancestor_core)
            #     population.append(
            #         self.indiv.initialize_chromosome(
            #             mutated_fuels_gnome,
            #             i+1
            #         )
            #     )
            population.sort(reverse=True, key=self._fitness_score)
        return population

    def search(
        self,
        generations:int = 100,
        max_generations:int = 500,
        ind_score:float = 0.95,
        population_score:float = 0.90
        
    ):
        
        #* population of chromosomes generated from given ancestor
        #* sorted by score
        population = self.make_population()
        
        score_population = self._get_population_average(population, 'fitness_score')
        print(
            f"start search with populations score: {score_population}"
            + f"and burnup {self._get_population_average(population, 'core_burnup')}"
        )

        search = True
        generation = 1
        try:
            while search:

                if population[0]["fitness_score"] >= ind_score \
                    or score_population >= population_score:
                    print(population[0]["fitness_score"])
                    break
                elif generations >= max_generations:
                    break
                elif generation == generations and generations < max_generations:
                    generations += 10
                
                st_time = time.time()
                #* saving best solution
                self._best_per_iter.append(
                    population[0].copy()
                )

                self._aver_score_per_iter.append(
                    self._get_population_average(population, 'fitness_score')
                )

                new_generation = []
                population_burnup = self._get_population_average(population, "core_burnup")

                # #* shows how ids distibuted in population
                # population_chromosomes_id = list(map(lambda x: x["id"], population))

                best_fit_number = math.ceil(self.elitism*len(population))
                # print(f"{best_fit_number} goes next")
                # print(population[:best_fit_number])
                #* population of new_generation array
                new_generation.extend(population[:best_fit_number])

                #* drop chromosomes that goes next round
                population = population[best_fit_number: ]

                #*** matting
                mate_size = math.ceil(self.mate_probability*len(population))
                if not mate_size % 2 == 0:
                    mate_size += 1
                
                #* choices method allows repetitions in return
                #* list is shuffled
                to_mate_chromosomes = random.choices(
                    population,
                    k=mate_size
                )
                parents_split = int(mate_size/2)
                
                #todo can be speeded up
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    ps = tuple(zip(
                        to_mate_chromosomes[ :parents_split], 
                        to_mate_chromosomes[parents_split: ]
                    ))
                    ps_futures = [
                        executor.submit(self.mate, p1.copy(), p2.copy()) for p1, p2 in ps
                    ]
                    
                    for ps_future, (p1, p2) in zip(concurrent.futures.as_completed(ps_futures), ps):
                        offspring = ps_future.result()
                        # print(offspring)
                        
                        with lock:
                            if offspring is None:
                                continue
                            
                            
                            family = [
                                p1.copy(),
                                p2.copy(),
                                offspring    
                            ]
                            
                            best_in_family = self.mate_tournament_selection(family)
                            
                            population = self._replace_chromosomes(
                                population,
                                best_in_family
                            )
                #todo 

                # for p1, p2 in zip(
                #     to_mate_chromosomes[ :parents_split], 
                #     to_mate_chromosomes[parents_split: ]
                # ):
                #     #* copies to prevent overwriting real objects
                #     offspring = self.mate(
                #         p1.copy(),
                #         p2.copy()
                #     )
                #     if offspring is None:
                #         continue
                    
                #     family = [
                #         p1.copy(),
                #         p2.copy(),
                #         offspring    
                #     ]
                    
                #     best_in_family = self.mate_tournament_selection(family)
                    
                #     population = self._replace_chromosomes(
                #         population,
                #         best_in_family
                #     )
                
                #*** permutation mutation part
                permutation_mutation_number = math.ceil(
                    self._initilize_mutation_probability(generation)
                    *len(population)
                )

                to_permutate_chromosomes = random.sample(
                    population,
                    k=permutation_mutation_number
                )

                for chromo in to_permutate_chromosomes:
                    mutated_chromo = self.permutation_mutation(chromo)
                    population = self._replace_chromosomes(
                        population,
                        mutated_chromo
                    )

                #* fresh fuel mutation part
                fresh_fuel_mutation_number = math.ceil(
                    self._initilize_fresh_fuel_probability_exp(population_burnup)
                    *len(population)
                )

                to_fresh_fuel_mutation = random.sample(
                    population,
                    k=fresh_fuel_mutation_number
                )

                for chromo in to_fresh_fuel_mutation:
                    fresh_chromo = self.fresh_fuel_mutation(chromo)
                    population = self._replace_chromosomes(
                        population,
                        fresh_chromo
                    )
                
                new_generation.extend(population)
                population = new_generation

                population.sort(reverse=True, key=self._fitness_score)

                score_population = self._get_population_average(population, 'fitness_score')
                burnup_population = self._get_population_average(population, 'core_burnup')
                print(f"average populations score {score_population}"
                + f" and burnup {burnup_population} at the end of {generation} generation")
                

                generation += 1

                fn_time = time.time() - st_time

                print(fn_time)

        except KeyboardInterrupt:
            #* returns population whenever KeyboardInterrupt raises
            return population    
        
        return population

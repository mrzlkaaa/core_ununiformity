from operator import concat
from . import  *
import pandas as pd
import numpy as np

try:
    from .core import Cartogram
except ImportError:
    from core import Cartogram


class Permutations(Cartogram):
    SIX_TUBE = 265
    EIGHT_TUBE = 300


    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        ind:int | None = None,
        name: str | None = None,
    ):
        
        if ind is not None and name is None:
            df = df.loc[ind:ind, :]
            if not len(df) > 0:
                raise ValueError("Row with a given index does not exist")
        elif name is not None and ind is None:
            df = df[df["Name"] == name]
        elif name is None and ind is None:
            raise ValueError("Both ind and name arguments are None. Provide one")
        elif name and ind:
            raise ValueError("Both ind and name arguments are given. Choose one ")

        
        super().__init__(df)
        self.df_dump = self.df.copy()
        print(self.df_dump.columns)
        self.interchan_8th, self.interchan_6th = self._quick_examination_processing()

    def _quick_examination_processing(self):
        '''
        #* Internal method that runs at initialization step
        #* The method checks for required columns in dataframe
        #* Also drops all columns indexed by _p and _s prefixes
        #* Map 6/8 tubes fuel positions and makes separated arrays to store cell names
        #* Parameters
        #* ----------
        #* None
        #* Raises
        #* ----------
        #* KeyError
        #*  Raises if required any of required column is not on DataFrame
        #* Returns
        #* ----------
        #* 2 arrays of columns names of 8 and 6 tubes fuel respectively
        '''
        try:
            for i in ["Name", self.CORE_PARTS["fuel_type"]["ALL_CELLS"], self.CORE_PARTS["burnup"]["ALL_CELLS"]]:
                self.df.loc[:, i]
        except KeyError:
            raise KeyError(f'some columns from {i} are not in DataFrame')
        

        #* preparation of arrays of interchangeable cells
        #* the idea is simple - cells with same inital mass of U235 are interchangeable
        res_6 = self.df.loc[:, self.CORE_PARTS["fuel_type"]["ALL_CELLS"]].apply(lambda x: x == self.SIX_TUBE) #
        res_8 = self.df.loc[:, self.CORE_PARTS["fuel_type"]["ALL_CELLS"]].apply(lambda x: x == self.EIGHT_TUBE) #.dropna(axis=1)

        
        #* splitting by prefix and mapping  
        res_6 = list(map(lambda x: x.split("_")[0], res_6[res_6 == True].dropna(axis=1).columns))
        res_8 = list(map(lambda x: x.split("_")[0], res_8[res_8 == True].dropna(axis=1).columns))
        
        
        return res_8, res_6

    def _cells_to_permutate_finder(
        self,
        cells: list | None = None
    ):
        df_ = self.df.copy()
        if cells is None:
            cells = self.CORE_PARTS["default"]["ALL_CELLS"]

        cells = list(set(self.interchan_8th).intersection(set(cells)))
        #* searcing among all columns
        all_cells = [j for i in cells  for j in self.df.columns if i in j]

        df_ = df_.loc[:, all_cells]

        #? independent method to filter by prefix
        selected_l = [i for i in all_cells if "_l" in i]
        selected_s = [i for i in all_cells if "_s" in i]
        
        
        coefs_ = np.argsort((df_.loc[:, selected_l].to_numpy() * df_.loc[:, selected_s].to_numpy()))
    
        return np.array(cells)[coefs_[0]][::-1]

    def _check_cells(
        self,
        from_: list,
        to_: list = []
    ):
        #? can be modified to decorator
        '''
        #* Internal method to check that given
        #* cells are interchangeable
        #* Note: Interchangeable cells are cells 
        #* with same fuel assembly type (8th -> 8th, 6th -> 6th)
        #* Parameters
        #* ----------
        #* from_: list
        #*  cells to permutate
        #* to_: list
        #*  cells with whose < from_>  cells will be permutated
        #* Raises
        #* ----------
        #* ValueError:
        #*  raises if any of given cell is non-interchangeable
        #* Returns
        #* ----------
        #* None
        '''
        concat_ = [*from_, *to_]
        
        #* check whether permutaion possible
        six_tube =list(set(self.interchan_6th).intersection(set(concat_)))
        eight_tube =list(set(self.interchan_8th).intersection(set(concat_)))

        print(
            "Detected among six tubes", six_tube, 
            "Detected among eight tubes", eight_tube
        )

        if concat_ is six_tube and concat_ is eight_tube:
            raise ValueError("Provided cells names are not suitable to make permutaions")
        
    def _columns_lookup(self, df:pd.core.frame.DataFrame, keys: list):
        return [j for i in keys  for j in df if i in j]

    
    def _get_swap_with(self, cell, drop_cell:bool=True):
         #* gettting type of fuel
        to_swap_with_ = self.interchan_8th.copy() if cell in self.interchan_8th\
            else self.interchan_6th.copy() if cell in self.interchan_6th\
            else ValueError(f"The cell {cell} has not been found among 8-th or 6-th FAs")
        
        if drop_cell:
            to_swap_with_.remove(cell)

        return  to_swap_with_
       
            

    def _random_permutations(
        self,
        number:int
    ):
        '''
        #* Implement permutation algorithm
        #* but every action is stored
        #* This is top level method for <make_permutations> and
        #* <add_fresh_fuel> methods
        #* In case if [7-6, 7-5] cells are given for
        #* fresh fuel installation, the <add_fresh_fuel>
        #* method called twice, so cells changed in sequential order
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
        cells = self.CORE_PARTS["default"]["ALL_CELLS"]

        from_ = []
        to_ = []
        for _ in range(number):
            i = np.random.randint(0,20) #* random int
            cell1 = cells[i]

            to_swap_with = self._get_swap_with(cell1)
            #! if _get_swap_with is successful clear commented below content
            # #* gettting type of fuel
            # to_swap_with = self.interchan_8th.copy() if cell1 in self.interchan_8th\
            #     else self.interchan_6th.copy() if cell1 in self.interchan_6th\
            #     else ValueError(f"The cell {cell1} has not been found among 8-th or 6-th FAs")

            # #* exclude chosen to permutate cell
            # to_swap_with.remove(cell1)
            
            j = np.random.randint(0, len(to_swap_with))
            cell2 = to_swap_with[j]

            from_.append(cell1)
            to_.append(cell2)

        return from_, to_
    
    def make_permutations(
        self,
        cells_from: list,
        cells_to: list,
        chain: bool = False,
        indep_chain: bool = False,
        merge:bool = False
        
    ):
        '''
        #* Cells permutations algorithm implemented
        #* data of < cells_from > swap with < cells_to > data
        #* Parameters
        #* ----------
        #* cells_from: list
        #*  cells to permutate
        #* cell_to: list
        #*  cells that will be swaped with < cells_from > cells
        #* chain: boolean
        #*
        #* indep_chain: boolean
        #*
        #* merge: boolean
        #*  works only if either indep_chain or chain is given    
        #* Raises
        #* ----------
        #* None
        #* Returns
        #* ----------
        #* copy of DataFrame where permutations are made
        #* or list of DataFrames
        '''
    
        self._check_cells(cells_from, cells_to)  #? can be used as decorator
        
        #* takes and stores values from <cells_to> cells
        #* replace values of <cells_to> by values from <cells_from>
        #* replace values of <cells_from> by values from <cells_to> (were stored)

        df = self.df.copy()
        to_drop_ = list(set(map(lambda x: x if "_s" in x or "_p" in x else None, df.columns)))
        df = df.drop(columns=to_drop_, errors="ignore")

        all_cells_to = self._columns_lookup(df, cells_to)
        all_cells_from = self._columns_lookup(df, cells_from)
        

        #* drop coef cells if cols are exists
        

        # print(self.df.loc[:, all_cells_to])
        # print(self.df.loc[:, all_cells_from])
        if not chain and not indep_chain:
            temp_storage = df.loc[:, all_cells_to]
            df.loc[:, all_cells_to] = df.loc[:, all_cells_from].values
            df.loc[:, all_cells_from] = temp_storage.values

            # print(self.df.loc[:, all_cells_to])
            # print(self.df.loc[:, all_cells_from])

            return df
        

        shape_ = (int(len(all_cells_to)/3), 3) #* new 2d shape
        #* make 2d array where each row assotiates with one cell
        all_cells_to = np.array(all_cells_to).reshape(shape_)
        all_cells_from = np.array(all_cells_from).reshape(shape_)

        dfs = []
        df_st = df.copy()
        for to_, from_ in zip(all_cells_to, all_cells_from):
            if indep_chain: #* prevent overwritting
                df_st = df.copy()
            
            temp_storage = df_st.loc[:, to_]
            df_st.loc[:, to_] = df_st.loc[:, from_].values
            df_st.loc[:, from_] = temp_storage.values
            dfs.append(df_st.copy())
        
        if merge:
            return pd.concat(
                dfs,
                axis=0
            )\
            .reset_index()\
            .drop("index", axis=1)

        return dfs

    
    def add_fresh_fuel(
        self, 
        cells: list,
        chain: bool = False
    ):
        '''
        #* Makes choosen cells 'fresh fuel-like'
        #* In other words, _b values are set to 0.0 -> fresh fuel
        #* Parameters
        #* ----------
        #* cells: list
        #*  names of columns where to "install" fresh fuel
        #* Raises
        #* ----------
        #* ValueError
        #*  raises if there are any other columns 
        #*  except with prefixes [ _b, _l, _t ]
        #* Returns
        #* ----------
        #* copy of DataFrame where permutations are made
        '''

        self._check_cells(cells)  #? can be used as decorator

        df = self.df.copy()
        to_drop_ = list(set(map(lambda x: x if "_s" in x or "_p" in x else None, df.columns)))
        df = df.drop(columns=to_drop_, errors="ignore")

         #* all cells with _t, _b and _l indexes
        all_cells = self._columns_lookup(df, cells)
        all_cells.sort() #* ascending order [ _b, _l, _t ]

        if not len(all_cells) % 3 == 0:
            raise ValueError("Each cell must containts 3 assotiated columns")

        shape_ = (int(len(all_cells)/3), 3) #* new 2d shape

        #* make 2d array where each row assotiates with one cell
        all_cells = np.array(all_cells).reshape(shape_)
        
        # print(all_cells)

        #* _b turns to zero, _l = _t
        dfs = []
        for i in all_cells:
            df.loc[:, i[0]] = 0.0
            df.loc[:, i[1]] = df.loc[:, i[2]]
            dfs.append(self.df.copy())
            
        if chain:
            return dfs
        
        return df


    def single_cell_pairwise_permutations(
        self,
        cell_from: str,
        pairwise_exceptions: list | None = None,
        make: bool = False,
        **kwargs, #* to pass some extra params
    ):
        '''
        #* Create all possible permutation pairs
        #* between given cell and other cells of the same fuel type
        #* If some cells should be excluded they shall be passed to
        #* < pairwise_exceptions > argument
        #* Parameters
        #* ----------
        #* cell_from: str
        #*  cell upon which permutations makes
        #* pairwise_exceptions: list
        #*  the list of cell that excludes from permutations pattern
        #* make: boolean
        #*  indicates whether < make_permutations > calls to make permutations
        #* **kwargs: dict
        #*  used to pass argumets for < make_permutations > method
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #* dict | [list, list]
        '''
        pairs = self._get_swap_with(cell_from)
        if pairwise_exceptions:
            to_exclude = set(pairwise_exceptions).intersection(set(pairs))

            pairs = [i for i in pairs if i not in to_exclude]

        cell_from_full = np.full((len(pairs), ), cell_from)
        
        if make:
            return self.make_permutations(
                cells_from=cell_from_full,
                cells_to=pairs,
                **kwargs
            ), cell_from_full, pairs

        return cell_from_full, pairs
        




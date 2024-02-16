import pandas as pd
from collections import defaultdict
try:
    from . import *
except ImportError:
    from main import *
import numpy as np



class Cartogram:

    R, C = 6,4
    R1, C1 = 2,2
    CORE_MAP = {
        "full": {
                "by_index": {
                0 : [0, 0],
                1 : [0, 1],
                2 : [0, 2],
                3 : [0, 3],
                4 : [1, 0],
                5 : [1, 1],
                6 : [1, 2],
                7 : [1, 3],
                8 : [2, 0],
                9 : [2, 3],
                10 : [3, 0],
                11 : [3, 3],
                12 : [4, 0],
                13 : [4, 1],
                14 : [4, 2],
                15 : [4, 3],
                16 : [5, 0],
                17 : [5, 1],
                18 : [5, 2],
                19 : [5, 3],
            },
            
            "by_name": {
                "7-6" : [0, 0],
                "7-5" : [0, 1],
                "7-4" : [0, 2],
                "7-3" : [0, 3],
                "6-6" : [1, 0],
                "6-5" : [1, 1],
                '6-4' : [1, 2],
                "6-3" : [1, 3],
                "5-6" : [2, 0],
                "5-3" : [2, 3],
                '4-6' : [3, 0],
                "4-3" : [3, 3],
                "3-6" : [4, 0],
                "3-5" : [4, 1],
                "3-4" : [4, 2],
                "3-3" : [4, 3],
                "2-6" : [5, 0],
                "2-5" : [5, 1],
                "2-4" : [5, 2],
                "2-3" : [5, 3],
            }
        },
        "quarter": {
                "by_index": {
                0 : [0, 0],
                1 : [0, 1],
                2 : [1, 0],
                3 : [1, 1],
                
            },
            
            "by_name": {
                "qul" : [0, 0],
                "qur" : [0, 1],
                "qll" : [1, 0],
                "qlr" : [1, 1],
            }
        },
        "adopted": {
            "by_index": {
                0 : [0, 0],
                1 : [0, 1],
                2 : [0, 2],
                3 : [0, 3],
                4 : [1, 0],
                5 : [1, 1],
                6 : [1, 2],
                7 : [1, 3],
                8 : [2, 0],
                9 : [2, 1],
                10 : [2, 2],
                11 : [2, 3],
                12 : [3, 0],
                13 : [3, 1],
                14 : [3, 2],
                15 : [3, 3],
                16 : [4, 0],
                17 : [4, 1],
                18 : [4, 2],
                19 : [4, 3],
            
            }
        }

    }

    #* move it to __init__ for creating unique
    #* config file
    

    def __init__(
        self, 
        df: pd.core.frame.DataFrame | None = None,
    ) -> None:
        self.df = df
        self.CORE_PARTS = init_core_parts()  #* dependency
        self._make_new_columns()


    def _make_static_data(
        self,
        block_name: str,
        prefix: str,
        force_overwrite: bool = False
    ):

        '''
        #* For internal use only
        #* Modify < core_parts.json >
        #* Appends new block with a give name and
        #* creates cells with a given prefix
        #* Parameters
        #* ----------
        #* block name: str
        #*  name of new block to append to < core_parts.json > file
        #* prefix: str
        #*  prefix that will be added to a default cells names
        #*  the pattern is "7-6" -> "7-6_prefix"
        #* Raises
        #* ----------
        #* KeyError 
        #*   throws if given block_name 
        #*   already in .json file top level key
        #* Returns
        #* ----------
        #* None
        '''

        new_block = dict()
        new_block[block_name] = defaultdict(list)

        if not self.CORE_PARTS.get(block_name) is None and not force_overwrite:
            raise KeyError(f"The block with name {block_name} already exists")
        
        _keys = self.CORE_PARTS["default"].keys()
        
        for i in _keys:
            cells = self.CORE_PARTS["default"][i]
            for j in cells:
                new_block[block_name][i].append(f"{j}_{prefix}")


        with open(core_parts, "r") as f:
            d = json.load(f)

        with open(core_parts, "w") as f:
            d = {**d, **new_block}
            f.write(json.dumps(d, indent=4))
        
        
    def _make_new_columns(self):
        self.df.loc[
            :, 
            self.CORE_PARTS["left"]["ALL_CELLS"]
        ] = self.df.loc[:, self.CORE_PARTS["fuel_type"]["ALL_CELLS"]].to_numpy() - self.df.loc[:, self.CORE_PARTS["burnup"]["ALL_CELLS"]].to_numpy()

        self.df.loc[
            :, 
            self.CORE_PARTS["percentage"]["ALL_CELLS"]
        ] = self.df.loc[:, self.CORE_PARTS["burnup"]["ALL_CELLS"]].to_numpy() / self.df.loc[:, self.CORE_PARTS["fuel_type"]["ALL_CELLS"]].to_numpy() * 100

    
    def quick_processing(
        self, 
        process:str = "all"
    ):
        '''
        #* Quick processing of df that consists of
        #* reactor core data
        #* Method is used to create new features for
        #* burnup and ununiformity data 
        #* Parameters
        #* ----------
        #* ununiformity: bool
        #*  indicate whether ununiformity data reuqires
        #*  processing    
        #* Returns
        #* ----------
        #* None cuz method makes changes directly to class instance variable (self.df)
        '''
        # self.df = self.df.astype(float)

        to_process = [
            ("average", "ALL_CELLS"),
            ("right_center_side", "RIGHT_CENTER_SIDE"),
            ("left_center_side", "LEFT_CENTER_SIDE"),
            ("right_side", "RIGHT_SIDE"),
            ("left_side", "LEFT_SIDE"),
            ("center", "CENTER"),
            ("qul", "QUL"),
            ("qur", "QUR"),
            ("qll", "QLL"),
            ("qlr", "QLR")
        ]
        exclude = []

        match process:
            case "nos":
                exclude = [
                    "RIGHT_SIDE", "LEFT_SIDE"
                ]

            case "noq":
                exclude = [
                    "QUL", "QUR", "QLL", "QLR"
                ]

            case "nosq":
                exclude = [
                    "RIGHT_SIDE", "LEFT_SIDE",
                    "QUL", "QUR", "QLL", "QLR"
                ]

        #* filter to_process by exclude elems
        if len(exclude) != 0:
            for i in exclude:
                for n,j in enumerate(to_process):
                    if to_process[n][1] == i:
                        to_process.remove(j)

        for k in self.CORE_PARTS.keys():
            
            for i, j in to_process:
                
                prefix_ = self.CORE_PARTS.get(k).get("prefix")
                columns_ = self.CORE_PARTS.get(k).get(j)
                try:
                    #* making new features
                    self.df[
                        f"{i}{prefix_}"
                        ] = self.df.loc[:, columns_].mean(axis=1)
                except KeyError:
                    print(f"The columns with prefix {prefix_} are not in DataFrame")
                    break


    @staticmethod
    def from_arr_to_df(
        arr: np.ndarray,
        isdefaults: bool=True,
        custom_pos: list | None = None,
        data_type: str = "burnup"

    ):
        if isinstance(arr, list):
            arr = np.asarray(arr)

        if arr.shape[-1] > 1:
            arr = arr.reshape(-1)

        if isdefaults:
            cols = Cartogram.CORE_PARTS[data_type]["ALL_CELLS"]
        
        elif not isdefaults and isinstance(custom_pos, list):
            cols = custom_pos

        else:
            raise TypeError("custom_pos variable must be a list")

        return pd.DataFrame(data=arr, index=cols).T

    @classmethod
    def from_array(
        cls, 
        arr,
        **kwargs
    ) -> object:

        print(arr)
        df = cls.from_arr_to_df(arr=arr, **kwargs)
        return cls(df=df)





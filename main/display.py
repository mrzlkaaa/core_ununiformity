import pandas as pd
import numpy as np

from select import select

from typing import Union

try:
    from .core import Cartogram
except ImportError:
    from core import Cartogram

D = Union[np.ndarray, pd.core.frame.DataFrame, int]



class Display(Cartogram):
    COLUMNS_F = ["6", "5", "4", "3"]
    ROWS_F = ["7", "6", "5", "4", "3", "2"]

    COLUMNS_Q = ["1", "2"]
    ROWS_Q = ["1", "2"]

    def __init__(
        self,
        df: pd.core.frame.DataFrame | None = None,
    ):  
        super().__init__(df)
        self.cmap = "RdYlGn_r"
        

    def _show_gradient_cmap(
        self,
        df: pd.core.frame.DataFrame,
        # cmap: str,
    ):
        return df.style.background_gradient(cmap=self.cmap, axis=None)

    def _init_display(self, mode: str):
        match mode:
            case "f":
                config = self.CORE_MAP.get("full")
                core = np.full((self.R, self.C), np.nan)
                # core = np.full((self.R, self.C), "Be") #* color map is not avaliable that way
                rows, cols = self.ROWS_F, self.COLUMNS_F
            case "q":
                config = self.CORE_MAP.get("quarter")
                core = np.full((self.R1, self.C1), np.nan) 
                # core = np.full((self.R1, self.C1), "Be") #* color map is not avaliable that way
                rows, cols = self.ROWS_Q, self.COLUMNS_Q

        return config, core, rows, cols

    def display(
        self,
        data: D | None = None,
        ind: int | None = None,
        sel_cols: list | None = None,
        mode: str = "f",
        dt: str = "burnup",
        show_cmap: bool = True
    ) -> pd.core.frame.DataFrame:
        '''
        #* The method displays cartogram
        #* with background gradient
        #* Can be used to display burnup / uniniformity coefs
        #* Full, Quarter of core can be submitted to a method to be displayed
        #* The method uses submethods to apply display feature on a given numpy array
        #* and pandas Series
        #* Parameters
        #* ----------
        #* data: D | None = None
        #*  Given data to display. Can be uses if df 
        #*  had not been passed during instance initialization 
        #* ind: int | None = None
        #*  can be used to select row from self.df
        #*  <works only with sel_cols and self.df>
        #* sel_cols: list | None = None
        #*  can be used to select columns from self.df
        #*  <works only with sel_cols and self.df>
        #* tp: str
        #*  tp stands for type of cartogram to display
        #*  "f" - full core, "q" - quarter of core
        #* dt: str
        #*  dt stands for data type and implies
        #*  what given data is
        #*  "burnup" - burnup cartogram, else / "coef" - uniniformity coefs
        #* show_cmap: boolean
        #*  if set to True displays data with gradient background
        #* Raises
        #* ----------
        #*  TypeError - if a given data has a type 
        #* Returns
        #* ----------
        #* df
        '''
        # self.core_full = np.full((self.R, self.C), np.nan)
        # self.core_quarter = np.full((self.R1, self.C1), np.nan)

        # config = {}

        config, core, rows, cols = self._init_display(mode)
        dt_prefix = "" if dt == "burnup" else dt

        if type(data) == np.ndarray:
            config = config.get("by_index")
            self.as_arr(
                data=data,
                template=core,
                config=config
            )

        elif type(data) == pd.core.series.Series:
            config = config.get("by_name")
            self.as_series(
                data=data,
                template=core,
                config=config,
                prefix=dt_prefix
            )
        elif data is None and ind is not None and sel_cols is not None:
            config = config.get("by_name")
            df = self.as_series(
                data=self.df.loc[ind, sel_cols],
                template=core,
                config=config,
                prefix=dt_prefix
            )
        else:
            raise TypeError("Given data has a data type that cannot be processed yet")

        df = pd.DataFrame(data=core, index=rows, columns=cols)

        if show_cmap:
            return self._show_gradient_cmap(df)
        return df

    def as_series(
        self,
        data: pd.core.series.Series,
        template: np.ndarray,
        config: dict,
        prefix: str,
    ) -> None:
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


        for n in config.keys():
            template[
                config[n][0]][config[n][1]
            ] = data.loc[n+prefix]

    def as_arr(
        self,
        data: np.ndarray,
        template: np.ndarray,
        config: dict,
    ) -> None:

        for n in range(len(data)):
            template[
                config[n][0]][config[n][1]
            ] = data[n]
        
        
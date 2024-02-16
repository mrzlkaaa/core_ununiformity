from distutils import core
import pandas as pd
import numpy as np
import json

from collections import defaultdict


core_parts = "core_parts.json"

__all__ = [
    "json", "init_core_parts", "core_parts", "unpack_stacking_model"
]

def init_core_parts():
    with open(core_parts, "r") as f:
         return json.load(f)
        
def unpack_stacking_model(
    models:dict,
    X_val_raw,
):
    res_dict = defaultdict(list)
    for key, model in models.items():
        # res_dict[key] = defaultdict()
        for k, v in model.items():
    
            base = v["base"]
            X_val = base["ct_base"].transform(X_val_raw)
            X_val_base = X_val.copy()
            
        
            # print(X_val_raw)
            for kb, bvals in base["models"].items():
                if kb == "bagging" and len(bvals) > 0:
                    for bag_names, bag_mods in base["models"].get("bagging").items():
                        for bnum, bag_mod in enumerate(bag_mods):
                            X_val_base.loc[:, [f"{bag_names}_{bnum}"]] = bag_mod.predict(X_val)
                
                elif kb != "bagging":
                    X_val_base.loc[:, kb] = bvals.predict(X_val)
                
            X_meta = X_val_base

            # X_meta = X_meta.loc[
            #     :,
            #     list(set(X_meta).difference(set(X_val)))
            # ]
            
            X_meta = X_meta.reindex(sorted(X_meta), axis=1)

            res_dict[k].append( v["meta"].predict(X_meta) )
    
    res_arr = np.array(list(res_dict.values())).mean(axis=1)
    
    res = pd.DataFrame(data=res_arr).T
    res.columns = res_dict.keys()
    
    return res
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import json
import read_cloks as rqe 
import glob 


# %%
def get_stuff(fname, algoname='davidson', other_info=None):
    dims= rqe.read_dimensions(fname)
    if dims is None:
        return None
    nk = rqe.read_nkpoints(fname)
    if nk is None:
        return None
    dims.update({'nkpoints':nk})
    para  = rqe.read_parallel(fname)
    if para is None:
        return None
    try:
        clocks = dict(rqe.read_clocks(fname))
        iterations=  dict(rqe.read_iterations(fname))
    except TypeError:
        return None
    raminfo = rqe.read_raminfo(fname)
    data1 = {"output":fname, 'algo':algoname}
    data1.update({'clocks':clocks}) 
    data1.update({'iter':iterations})
    para
    dims.update(para)
    data1.update ({'dims':dims})
    if other_info is not None:
        data1.update(other_info) 
    if raminfo is not None:
        data1.update({"RAM":raminfo})
    return data1 


# %%
other_info = {
    "CPU":"Intel Xeon 8160 CPU @ 2.10GHz",
    "Node":"2*24-core",
    "Memory": "192 GB DDR4 RAM",
    "Net": "Intel OmniPath (100Gb/s) high-performance network"
}
pathre='../acque/aqua*/out_*'
jsonname='acque.json'
data=(get_stuff(name, other_info= other_info) for name in glob.glob('pathre'))
#
data_good= [_ for _ in filter(None, data)]

with open (jsonname,'w') as fw:
    json.dump(data_good, fw, indent=2)
len(data_good) 


# %%

from collections import deque
d = deque(filter(lambda _: _['clocks']['electrons']>_['clocks']['PWSCF'], data_good))


# %%
d


# %%
data_good[0]['clocks']['PWSCF']


# %%




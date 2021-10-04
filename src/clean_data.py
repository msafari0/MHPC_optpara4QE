import json
import pandas as pd
import numpy as np
import collections.abc
from ase.data import atomic_numbers

elements = list(atomic_numbers.keys())


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if type(v) == list:
                if type(v[1]) == str:
                    items.append((new_key, v[0]))
                    items.append((new_key + "_unit", v[1]))
                else:
                    for it, v_ in enumerate(v):
                        items.append((new_key + f"_{it}", v_))

            else:
                items.append((new_key, v))
    return dict(items)


def compute_nbetas(NL, NBETA, NATOMSTYPE):
    nl = np.array(NL.split('|')[1::2], dtype='int')
    betas_per_elements = np.array(NBETA.split('|')[1::2], dtype='int')
    natom_per_element = np.array(NATOMSTYPE.split('|')[3::4], dtype=np.float)
    nelements = len(betas_per_elements)
    natoms_list = [np.repeat(natom_per_element[i], betas_per_elements[i])
                   for i in range(nelements)]
    natoms_list = np.array(
        [item for sublist in natoms_list for item in sublist])
    try:
        tot_betas = np.sum(2*nl+1 * natoms_list)
    except:
        return np.sum(2*nl+1 * natom_per_element[0])
    return tot_betas


def count_lanth_trans(data):
    lanthanid = np.zeros(len(data))
    transition = np.zeros(len(data))

    lanthanids = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                  'Gb', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    first_row_transition_metals = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni']

    for el in lanthanids:
        lanthanid += [int(el in line.split('|')[1::4])
                      for line in data['NatomsType']]
    lanthanid = np.array(lanthanid)  # .astype('bool').astype('int')
    for el in first_row_transition_metals:
        transition += [int(el in line.split('|')[1::4])
                       for line in data['NatomsType']]
    transition = np.array(transition)  # .astype('bool').astype('int')

    return lanthanid, transition


def clean_marconi_data(filename, architecture_id=0):
    with open(filename) as f:
        data = json.load(f)

    cname = 'computational_complexity'
    tname = 'time_per_call'
    norm_tname = 'normalized_time_per_call'

    # Flatten all nested dictionaries
    data = [flatten_dict(d) for d in data]

    df = pd.DataFrame.from_dict(data).drop(0)
    df = df.rename(columns={'dims_Threads': 'threads_per_node'})

    lanthanid, transition = count_lanth_trans(df)
    df['n_lanthanid'] = lanthanid
    df['n_transition'] = transition
    df['n_betas'] = [compute_nbetas(nl, nb, na) for nl, nb, na in zip(
        df['Nl'], df['Nbeta'], df['NatomsType'])]
    # df = df.drop(index=738)  # Remove an outlier

    # The first column is alwyas the target
    columns_to_keep = [norm_tname, 'iter_sum_band', tname, 'n_el',
                       'n_el^3', 'n_species', 'dims_nat', 'n_transition',
                       'n_lanthanid', 'dims_nbands', 'convergence',
                       'smooth_grid_rec', 'dims_nkpoints', 'n_betas',
                       'n_cores', 'n_nodes', 'threads_per_node', 'dims_npool']

    if df['Node'].values[0] == '2*24-core':
        corespernode = 48
    elif df['Node'].values[0] == '32-core':
        corespernode = 32
    df_ = pd.DataFrame()
    for c in list(df.columns):
        if len(df[c].unique()) > 1 or c in columns_to_keep:
            try:
                df_[c] = df[c].astype('float')
            except ValueError:
                pass

    # The column we use as time is the time per call of the iteration
    df_[tname] = df_['clocks_PWSCF']*(1/df_['iter_sum_band'])

    df_['n_cores'] = df_['threads_per_node']*df_['dims_MPI tasks']
    df_['n_nodes'] = df_['n_cores']//corespernode
    df_['n_el^3'] = df_['n_el']**3
    df_[cname] = df_['n_el^3']*df_['dims_nkpoints']/df_['n_cores']
    df_[norm_tname] = df_[tname]/df_[cname]



    df_tot = pd.DataFrame()
    for c in columns_to_keep:
        df_tot[c] = df_[c]

    df_tot = df_tot.rename(columns={'iter_sum_band': 'n_calls',
                                    'dims_npool': 'n_pool',
                                    'dims_nkpoints': 'n_k',
                                    'dims_nat': 'n_at',
                                    'dims_nbands': 'n_ks',
                                    'smooth_grid_rec': 'n_g_smooth'})

    df_tot['arch'] = architecture_id

    df_el = pd.DataFrame(index=df_tot.index, columns=elements)
    df_el.loc[df_el.index, elements] = 0
    df_el[df.pseudo.str.get_dummies().columns] = df.pseudo.str.get_dummies().values

    df_tot[elements] = df_el
    del df_el, df

    return df_tot


def clean_chemistry_data(filename, column_order, architecture_id=2):

    data = pd.read_csv(filename)

    cname = 'computational_complexity'
    tname = 'time_per_call'
    norm_tname = 'normalized_time_per_call'
    data = data[data['smoothgrid'] != "E"]
    data = data.dropna(subset=['timer'])
    data = data[data['timer'].astype(np.float) < 15000]
    data = data[data['timer'].astype(np.float) > 20]
    data['time_per_call'] = data['timer']/data['calls']

    # Define the features for the learning algorithm
    timer = np.array([np.float(datum_curr)
                      for datum_curr in data['time_per_call']])

    lanthanid, transition = count_lanth_trans(data)

    features = np.stack((2 - np.array(data['NDiag'] == 'serial', dtype=np.float),
                         [len(line.split('|')[1::2])
                          for line in data['pseudo']],
                         (np.array(data['NBgrp'], dtype=np.float)),
                         (np.array(data['NPool'], dtype=np.float)),
                         (np.array(data['NCores'], dtype=np.float)),
                         (np.array(data['Nk'], dtype=np.float)),
                         (np.array(data['NKS'], dtype=np.float)),
                         (np.array(data['Nelectrons'], dtype=np.float)),
                         (np.array(data['convergence'], dtype=np.float)),
                         (np.array(data['NThreads'], dtype=np.float)),
                         (np.array(data['calls'], dtype=np.float)),
                         ([np.sum(np.array(line.split('|')[3::4], dtype=np.float))
                           for line in data['NatomsType']]),
                         ([compute_nbetas(nl, nb, na) for nl, nb, na in zip(
                             data['Nl'], data['Nbeta'], data['NatomsType'])]),
                         ([np.sum(2*np.array(line.split('|')[1::2], dtype=np.float)+1)
                           for line in data['Nl']]),
                         ([np.array(line.split('|')[1], dtype=np.float)
                           for line in data['densegrid']]),
                         ([np.prod(np.array(line.split('|')[3::2], dtype=np.float))
                           for line in data['densegrid']]),
                         ([np.array(line.split('|')[1], dtype=np.float)
                           for line in data['smoothgrid']]),
                         ([np.prod(np.array(line.split('|')[3::2], dtype=np.float))
                           for line in data['smoothgrid']]),
                         ([np.prod(np.array(line.split('|')[1::2], dtype=np.float))
                           for line in data['fftparinfo']]),
                         transition,
                         lanthanid
                         ), axis=-1)

    column_names = ['n_nodes', 'n_species', 'NBgrp', 'n_pool', 'n_cores',
                    'n_k',  'n_ks', 'n_el', 'convergence', 'n_threads',
                    'n_calls', 'n_at',
                    'tot_betas', 'unweighted_nl', 'dense_grid_rec',
                    'dense_grid_real', 'n_g_smooth',
                    'smooth_grid_real', 'fftparinfo',
                    'n_transition', 'n_lanthanid']

    df = pd.DataFrame(columns=column_names, data=features)
    df.loc[df.n_species == 0, 'n_species'] = 1
    df[tname] = timer
    df['n_el^3'] = df.n_el**3
    df[cname] = df['n_el^3']*df['n_k']/df['n_cores']
    df[norm_tname] = df[tname]/df[cname]
    df['threads_per_node'] = df['n_cores'] / \
        df['n_threads']/df['n_nodes']

    df.drop(columns=['fftparinfo', 'smooth_grid_real', 'dense_grid_real',
                     'computational_complexity', 'dense_grid_rec', 'NBgrp',
                     'unweighted_nl', 'n_threads'], inplace=True)

    df = df.rename(columns={'tot_betas': 'n_betas'})
    df['arch'] = architecture_id
    df = df[df.columns]

    df_el = pd.DataFrame(index=df.index, columns=elements)
    df_el.loc[df_el.index, elements] = 0
    df_el[data.pseudo.str.get_dummies(
    ).columns] = data.pseudo.str.get_dummies().values

    df[elements] = df_el
    del df_el

    return df

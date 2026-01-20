from collections import deque, Counter
import json
import glob


def tot_string(s):
    return s.split('CPU')[-1].replace('WALL', '').strip().replace(' ', '')


def read_betas(fname):
    with open(fname, 'r') as f:
        l_ = f.readlines()

    # Identify the indices for beta sections
    index_start = [i + 1 for i, line in enumerate(l_) if 'Using radial grid of ' in line]
    index_end = []
    
    # Dynamically determine the end markers for each block
    for start_idx in index_start:
        # Look for the next 'Q(r) pseudized with' or next 'PseudoPot.' as a fallback
        end_idx = next(
            (i for i in range(start_idx, len(l_))
             if 'Q(r) pseudized with' in l_[i] or 'PseudoPot.' in l_[i]),
            len(l_)  # Default to end of file if no end marker is found
        )
        index_end.append(end_idx)

    nbetas = ""
    all_orbs = ""

    for s, e in zip(index_start, index_end):
        try:
            # Extract beta and orbital information safely
            nbeta = "|%s|" % (l_[s - 1].split()[6])
            orbs = [
                "|%s|" % (_.split()[2])
                for _ in l_[s:e]
                if len(_.split()) > 2  # Ensure the line has enough tokens
            ]
            all_orbs += "".join(orbs)
            nbetas += nbeta
        except IndexError as err:
            print(f"IndexError in file {fname} between lines {s} and {e}: {err}")
            print(f"Skipping problematic section.")
            continue
        except Exception as err:
            print(f"Unexpected error in file {fname} between lines {s} and {e}: {err}")
            raise

    return nbetas, all_orbs


def read_additional_info(fname):
    with open(fname, 'r') as f:
        l_ = f.readlines()
    natoms = [_.split()[-1] for _ in l_ if 'number of atoms/cell' in _][0]
    species = [_.split()[1] for _ in l_ if 'tau(' in _]
    species = species[:int(natoms)]
    n_spec = dict(Counter(species))
    species_line = ["|%s||%i|" % (k, v) for k, v in n_spec.items()]
    species_line = "".join(species_line)
    print(species_line)
    psuedo_line = ["|%s|" % (k) for k in n_spec.keys()]
    psuedo_line = "".join(psuedo_line)
    nel = [_.split()[-1] for _ in l_ if 'number of electrons' in _][0]
    conv_thresh = [_.split()[-1]
                   for _ in l_ if 'convergence threshold' in _][0]
    n_g_smooth = [_.split()[2] for _ in l_ if 'Smooth grid:' in _][0]
    n_g_dense = [_.split()[2] for _ in l_ if 'Dense  grid:' in _][0]
    return (nel, species_line, len(n_spec),
            n_g_smooth, n_g_dense, conv_thresh, psuedo_line)


def read_tot(s):
    t = s.partition('s')[0]
    part = t.partition('m')
    if part[1] == 'm':
        try:
            res = float(part[2])
        except ValueError:
            res = 0.
        t = part[0]
        part = t.partition('h')
        if part[1] == 'h':
            res = res + 60*(float(part[2])+60*float(part[0]))
        else:
            res = res+60*float(part[0])
    else:
        res = float(part[0])
    return res


def read_clocks(filename):
    with open(filename, 'r') as f:
        clocklines = deque(filter(lambda s: ' CPU ' in s and ' WALL' in s, f))
    if len(clocklines) == 0:
        return None
    for prog in ['PWSCF', 'CP ']:
        totclock = deque(filter(lambda s: prog in s, clocklines), maxlen=1)
        if len(totclock) == 1:
            # res = ((prog, read_tot(totclock[0].split()[-2])),)
            res = ((prog, read_tot(tot_string(totclock[0]))),)
            break
    clocks = [
        (_.split()[0], float(_.split()[4].replace('s', '')))
        for _ in clocklines
        if ('PWSCF' not in _) and ('CP ' not in _)]
    return tuple(clocks)+res


def read_iterations(filename):
    with open(filename, 'r') as f:
        clocklines = [_ for _ in f.readlines(
        ) if ' CPU ' in _ and ' WALL' in _]
    iterations = [
        (_.split()[0], float(_.split()[7].replace('s', '')))
        for _ in clocklines
        if ('PWSCF' not in _) and ('CP ' not in _)]
    return tuple(iterations)


def read_program(filename):
    with open(filename) as f:
        startline = [_ for _ in f.readlines(
        ) if 'Program' in _ and 'starts' in _][0]
    return startline.split()[1]


def read_ndiag(line):
    ll = line.split('*')[1]
    ndiag = int(ll.split()[0])
    return ndiag*ndiag


def read_parallel(filename):
    with open(filename, 'r') as f:
        l_ = f.readlines()[:50]
    try:
        linetoread = [_ for _ in l_ if 'Number of MPI processes:' in _][0]
    except IndexError:
        return None
    res = {'MPI tasks': int(linetoread.split()[4])}
    linetoread = [_ for _ in l_ if 'Threads/MPI process:' in _][0]
    res.update({'Threads': int(linetoread.split()[2])})
    try:
        linetoread = [_ for _ in l_ if "K-points division:     npool" in _][0]
        res.update({'npool': int(linetoread.split()[-1])})
    except IndexError:
        res.update({'npool': 1})
    try:
        linetoread = [_ for _ in l_ if "R & G space division:" in _][0]
        r_n_g = int(linetoread.split()[-1])
        res.update({'n_RG': r_n_g})
    except IndexError:
        r_n_g = 1
    linetoread = [_ for _ in l_ if "wavefunctions fft division:" in _]
    if len(linetoread) == 0:
        wfc_fftdiv = (1, r_n_g)
    elif len(linetoread) > 0:
        wfc_fftdiv = tuple([int(_) for _ in (linetoread[0]).split()[-2:]])
    res.update({"wfc_fft_division": wfc_fftdiv})
    res.update({"taskgroups": len(linetoread) == 2})
    try:
        linetoread = [_ for _ in l_ if "distributed-memory algorithm" in _][0]
        res.update({'ndiag': read_ndiag(linetoread)})
    except IndexError:
        res.update({'ndiag': 1})
    return res


def read_gridinfo(filename, stringa):
    """
    filename: str path of the file to open
    stringa:  str string to search for selecting the line
    """
    with open(filename, 'r') as f:
        r = deque(filter(lambda _: stringa in _, iter(f)))[0]
    temp1, temp2 = r.split(":")[1], r.split(":")[2]
    grid_vecs = int(temp1.split()[0])
    temp2 = temp2.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    fft_dims = tuple((int(_) for _ in temp2.split()))
    return {"ngrid_vecs": grid_vecs, "fft_dims": fft_dims}


def read_dimensions(filename):
    with open(filename, 'r') as f:
        l_ = f.readlines()
    s = "number of atoms/cell"
    try:
        r = [_ for _ in l_ if s in _][0]
    except IndexError:
        return None
    res = {'nat': int(r.split()[-1])}
    s = "number of Kohn-Sham states="
    try:
        r = [_ for _ in l_ if s in _][0]
    except IndexError:
        return None
    res.update({'nbands': int(r.split()[-1])})
    s = "kinetic-energy cutoff"
    r = [_ for _ in l_ if s in _][0]
    res.update({'ecutwfc': float(r.split()[-2])})
    s = "charge density cutoff"
    r = [_ for _ in l_ if s in _][0]
    res.update({'ecutrho': float(r.split()[-2])})
    s = "unit-cell volume"
    r = [_ for _ in l_ if s in _][0]
    res.update({'vol': float(r.split()[-2])})
    dense_fft_dims = read_gridinfo(filename, "Dense  grid:")
    smooth_fft_dims = read_gridinfo(filename, "Smooth grid:")
    res.update({"Dense_grid": dense_fft_dims, "Smooth_grid": smooth_fft_dims})
    with open(filename, 'r') as f:
        r = deque(filter(lambda _: "Smooth grid:" in _, iter(f)))[0]
    return res


def read_raminfo(filename):
    total_ram = read_estimated_ram(filename)
    if total_ram is None:
        return None
    res = total_ram
    partial_ram = read_partial_ram(filename)
    if partial_ram is not None:
        res.update(partial_ram)
    return res


def read_estimated_ram(filename):
    with open(filename, 'r') as f:
        lines = [_ for _ in filter(
            lambda _: "Estimated" in _ and "RAM" in _, iter(f))]
    if len(lines) < 3:
        return None
    temp = lines[0].split('>')[1].split()
    static = (float(temp[0]), temp[1])
    temp = lines[1].split('>')[1].split()
    max_dynamic = (float(temp[0]), temp[1])
    temp = lines[2].split('>')[1].split()
    total = (float(temp[0]), temp[1])
    return {
        "static_per_process": static,
        "max_per_process": max_dynamic,
        "total": total
    }


def read_partial_ram(filename):
    with open(filename, 'r') as f:
        lines = [_ for _ in filter(lambda _:"Dynamical RAM for" in _, iter(f))]
    if len(lines) == 0:
        return None

    def read_line(_):
        temp1, temp2 = _.split(":")
        temp1 = temp1.replace("Dynamical RAM for", "").strip()
        return temp1, float(temp2.split()[0]), temp2.split()[1]
    itera = ((read_line(l)[0], read_line(l)[1:]) for l in lines)
    return dict(tuple(itera))


def read_nkpoints(fname):
    with open(fname, 'r') as fr:
        l_ = filter(lambda s: 'number of k points' in s, fr)
        l_ = deque(l_, maxlen=1)
    if len(l_) == 1:
        return int(l_[0].split()[4])
    else:
        return None


def get(fname, algoname='davidson', other_info=None):
    dims = read_dimensions(fname)
    if dims is None:
        # print("No dims for this file", fname)
        return None
    nk = read_nkpoints(fname)
    if nk is None:
        print("No k points for this file", fname)
        return None
    dims.update({'nkpoints': nk})
    para = read_parallel(fname)
    try:
        clocks = dict(read_clocks(fname))
        iterations = dict(read_iterations(fname))
    except TypeError:
        print("No Clock for this file", fname)
        return None
    raminfo = read_raminfo(fname)
    data1 = {"output": fname, 'algo': algoname}
    data1.update({'clocks': clocks})
    data1.update({'iter': iterations})
    dims.update(para)
    data1.update({'dims': dims})

    (nel, species_line, n_species, n_g_smooth,
     n_g_dense, conv_thresh, pseudo) = read_additional_info(
        fname)
    nbetas, all_orbs = read_betas(fname)
    data1.update({'Nbeta': nbetas})
    data1.update({'Nl': all_orbs})
    data1.update({'n_el': nel})
    data1.update({'n_species': n_species})
    data1.update({'NatomsType': species_line})
    data1.update({'pseudo': pseudo})
    data1.update({'smooth_grid_rec': n_g_smooth})
    data1.update({'dense_grid_rec': n_g_dense})
    data1.update({'convergence': conv_thresh})

    if other_info is not None:
        data1.update(other_info)
    if raminfo is not None:
        data1.update({"RAM": raminfo})
    return data1


def create_json(folder, inname="out_*", outname="data.json", other_info={
    "CPU": "Intel Xeon 8160 CPU @ 2.10GHz",
    "Node": "2*24-core",
    "Memory": "192 GB DDR4 RAM",
        "Net": "Intel OmniPath (100Gb/s) high-performance network"}):

    pathre = folder + inname
    data = (get(n, other_info=other_info) for n in glob.glob(pathre))
    data = [_ for _ in filter(None, data)]

    with open(outname, 'w') as fw:
        json.dump(data, fw, indent=2)

    return data

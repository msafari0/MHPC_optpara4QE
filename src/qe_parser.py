import re
import numpy as np, math
from pymatgen.core import Structure, Lattice
import spglib
import os
import math
from pymatgen.io.espresso.inputs.pwin import PWin as PWin
from math import ceil
from collections import defaultdict
#######################################################
# --- UPF pseudopotential parser and map to species ---
#######################################################

def build_species_to_pseudo_map_from_qeinput(qe_in, input_file_path, pseudos_dir):
    """
    Build a mapping: species_symbol -> dict with valence, nproj, and path.
    Example:
    {'Ti': {'valence': 12.0, 'nproj': 4, 'path': '/path/to/Ti.UPF'}, ...}
    
    Parameters:
    - qe_in: parsed QE input object
    - input_file_path: path to the QE input file
    - pseudos_dir: folder where pseudo files are stored (uploaded or local)
    """
    species_map = {}

    # 1) Try qe_in.atomic_species attribute
    try:
        atomic_species = getattr(qe_in, "atomic_species", None)
        if atomic_species:
            if isinstance(atomic_species, dict):
                for sym, entry in atomic_species.items():
                    pfile_name = None
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        pfile_name = entry[1].strip("'\"")
                    elif isinstance(entry, dict) and "pseudo" in entry:
                        pfile_name = entry["pseudo"].strip("'\"")
                    if pfile_name:
                        full_path = os.path.join(pseudos_dir, pfile_name)
                        val, nproj = parse_upf(full_path)
                        species_map[sym] = {"valence": val, "nproj": nproj, "path": full_path}
            else:
                for row in atomic_species:
                    try:
                        sym = row[0]
                        pfile_name = str(row[2]).strip("'\"")
                        full_path = os.path.join(pseudos_dir, pfile_name)
                        val, nproj = parse_upf(full_path)
                        species_map[sym] = {"valence": val, "nproj": nproj, "path": full_path}
                    except Exception:
                        pass
    except Exception:
        pass

    # 2) Fallback: parse ATOMIC_SPECIES block manually
    if not species_map:
        try:
            with open(input_file_path, "r") as fh:
                lines = fh.readlines()
            in_block = False
            for line in lines:
                l = line.strip()
                if l.upper().startswith("ATOMIC_SPECIES"):
                    in_block = True
                    continue
                if in_block:
                    if l == "" or l.startswith("&") or l.upper().startswith("K_POINTS") or l.upper().startswith("ATOMIC_POSITIONS"):
                        break
                    parts = l.split()
                    if len(parts) >= 3:
                        sym = parts[0]
                        pfile_name = parts[2].strip("'\"")
                        full_path = os.path.join(pseudos_dir, pfile_name)
                        val, nproj = parse_upf(full_path)
                        species_map[sym] = {"valence": val, "nproj": nproj, "path": full_path}
        except Exception as e:
            print(f"[WARN] Failed to parse ATOMIC_SPECIES block: {e}")

    return species_map



####################################
# --- UPF pseudopotential parser ---
####################################

def parse_upf(pseudo_file):
    """
    Parse UPF file to extract valence electrons and number of beta projectors.
    Works for UPF v1 and v2.
    """
    valence = None
    nproj = 0

    try:
        with open(pseudo_file, "r") as f:
            lines = f.readlines()

        content = "".join(lines)

        # --- UPF v2 / XML ---
        m_val = re.search(r"z_valence\s*=\s*\"?([\d\.]+)", content, re.IGNORECASE)
        if m_val:
            valence = float(m_val.group(1))

        m_proj = re.search(r"number_of_proj\s*=\s*\"?(\d+)", content, re.IGNORECASE)
        if m_proj:
            nproj = int(m_proj.group(1))

        # --- UPF v1 / text ---
        if valence is None:
            for line in lines:
                # Match number before "Z valence"
                m = re.search(r"([\d\.]+)\s+Z valence", line, re.IGNORECASE)
                if m:
                    valence = float(m.group(1))
                    break

        # --- UPF v1 / nproj fallback ---
        if nproj == 0:
            for line in lines:
                # match two numbers at line start, second number is nproj
                m = re.match(r"\s*\d+\s+(\d+)", line)
                if m:
                    nproj = int(m.group(1))
                    break

    except Exception as e:
        print(f"[parse_upf] Error reading {pseudo_file}: {e}")

    return valence, nproj
    
###############################
# --- Manual K_POINTS parser ---
###############################
def parse_kpoints_from_input(file_path):
    """
    Extract K_POINTS info manually from QE input.
    Supports 'automatic', 'gamma', and explicit lists.
    """
    mesh = None
    shift = [0, 0, 0]
    mode = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("K_POINTS"):
            tokens = line.split()
            if len(tokens) > 1 and tokens[1].lower().startswith("automatic"):
                mode = "automatic"
                # Next line has 6 integers: mesh(3), shift(3)
                parts = list(map(int, lines[i+1].split()))
                mesh = parts[:3]
                shift = parts[3:]
            elif len(tokens) > 1 and tokens[1].lower().startswith("gamma"):
                mode = "gamma"
                mesh = [1, 1, 1]
                shift = [0, 0, 0]
            else:
                mode = "explicit"
                try:
                    nk = int(lines[i+1].split()[0])
                except Exception:
                    nk = 1
                mesh = [nk, 1, 1]
                shift = [0, 0, 0]
            break

    return mode, mesh, shift

################################
# --- n_g_smooth extractting ---
################################

def compute_ng_smooth_cubic(celldm1_bohr, ecutwfc_Ry, ecutrho_Ry):
    """
    Compute n_g_smooth and density FFT dims for a cubic cell (ibrav=1).
    celldm1_bohr : lattice constant in BOHR
    ecutwfc_Ry, ecutrho_Ry : energy cutoffs in Rydberg
    Returns: (n_g_smooth, fft_dims_tuple)
    """
    a = float(celldm1_bohr)         # Bohr
    b = 2.0 * math.pi / a           # reciprocal-unit (1/Bohr)
    # index-space radius for wavefunctions
    R_wfc = math.sqrt(float(ecutwfc_Ry)) / b
    M_wfc = 2 * math.ceil(R_wfc)    # wavefunction FFT grid (even)
    M_rho = int(2 * M_wfc)          # density FFT grid
    
    # Nmax = threshold on n^2 (integer index squared sum)
    Nmax = float(ecutrho_Ry) / (b * b)
    lim = M_rho // 2
    
    # count integer grid points inside sphere, but only over the finite FFT cube
    full_count = 0
    independent_count = 0
    for n1 in range(-lim, lim):
        for n2 in range(-lim, lim):
            for n3 in range(-lim, lim):
                s = n1*n1 + n2*n2 + n3*n3
                if s <= Nmax + 1e-12:
                    full_count += 1
                    # canonical representative for Hermitian symmetry:
                    if (n1 > 0) or (n1 == 0 and n2 > 0) or (n1 == 0 and n2 == 0 and n3 >= 0):
                        independent_count += 1

    return independent_count, (M_rho, M_rho, M_rho)


def compute_ng_smooth_general(lattice_matrix, ecutwfc_Ry, ecutrho_Ry):
    """
    lattice_matrix : 3x3 matrix of direct lattice vectors in BOHR (rows or columns consistent)
    Returns: (n_g_smooth, fft_dims_tuple)
    """
    A = np.array(lattice_matrix, dtype=float)   # direct cell (units: BOHR)
    # reciprocal basis vectors (as columns)
    B = 2.0 * math.pi * np.linalg.inv(A).T     # columns are b1,b2,b3 (units 1/BOHR)

    # To estimate wavefunction FFT grid along each axis: find max integer index along axis i
    # We do a conservative isotropic estimate using scalar Gmax from ecutwfc:
    Gmax_wfc = math.sqrt(float(ecutwfc_Ry))   # in 1/BOHR units (see derivation)
    # choose a grid size M such that integer multipliers up to ceil(Gmax/|b_i|) are representable
    b_norms = np.linalg.norm(B, axis=0)  # norm of each reciprocal basis vector
    R_indices = [math.ceil(Gmax_wfc / bn) for bn in b_norms]
    M_wfc_axes = [2 * r for r in R_indices]
    # make them integers and even
    M_wfc_axes = [int(m if m%2==0 else m+1) for m in M_wfc_axes]
    # pick conservative uniform density grid as double the max wavefunction axis size
    M_rho = 2 * max(M_wfc_axes)
    lims = [M_rho//2] * 3

    # Now enumerate indices in limited cube and test actual |G|^2 <= ecutrho_Ry
    Nmax = float(ecutrho_Ry)
    full = 0
    independent = 0
    for n1 in range(-lims[0], lims[0]):
        for n2 in range(-lims[1], lims[1]):
            for n3 in range(-lims[2], lims[2]):
                n = np.array([n1, n2, n3], dtype=float)
                Gvec = B.dot(n)   # G in 1/BOHR
                G2 = float(Gvec.dot(Gvec))
                # In Rydberg units condition is G2 <= ecutrho_Ry (see unit mapping)
                if G2 <= Nmax + 1e-12:
                    full += 1
                    if (n1 > 0) or (n1 == 0 and n2 > 0) or (n1 == 0 and n2 == 0 and n3 >= 0):
                        independent += 1
    return independent, (M_rho, M_rho, M_rho)



####################################
# --- Main QE input parser 8 Oct ---
####################################

def parse_qe_input(input_file, pseudo_map=None, pseudos_dir=None):
    """
    Parse QE input file (using PWin + custom k-point parser) plus optional pseudo files.
    Returns a dict with key features for NN prediction, including:
    - total_valence
    - total_betas
    """
    if not isinstance(input_file, str):
        raise ValueError(f"Expected input_file to be path str, got {type(input_file)}")

    # Parse input with PWin
    qe_in = PWin.from_file(input_file)
    cell = qe_in.structure

    # Build Structure
    lattice = Lattice(cell.lattice.matrix)
    species = [str(s) for s in cell.species]
    frac_coords = cell.frac_coords
    structure = Structure(lattice, species, frac_coords)

    # ------------------------
    # Build species -> pseudo info map
    # ------------------------
    species_to_pseudo = build_species_to_pseudo_map_from_qeinput(qe_in, input_file, pseudos_dir)

    # Now compute total_valence and total_betas by summing over structure sites
    total_valence = 0.0
    total_betas = 0
    missing_pseudos = defaultdict(int)

    for site in structure:
        sym = site.species_string
        pseudo_info = species_to_pseudo.get(sym)
        val, nproj = None, None

        if pseudo_info:
            val = pseudo_info.get("valence")
            nproj = pseudo_info.get("nproj")
            # fallback: if not present, parse directly from path
            if (val is None or nproj is None) and "path" in pseudo_info:
                val, nproj = parse_upf(pseudo_info["path"])
        else:
            # Try to match by element name in pseudo_map
            if pseudo_map:
                for k in pseudo_map.keys():
                    if k.lower().startswith(sym.lower()):
                        val = pseudo_map[k].get("valence")
                        nproj = pseudo_map[k].get("nproj")
                        break
            # fallback: parse from pseudos_dir if available
            if (val is None or nproj is None) and pseudos_dir:
                for fname in os.listdir(pseudos_dir):
                    if fname.lower().startswith(sym.lower()):
                        full_path = os.path.join(pseudos_dir, fname)
                        val, nproj = parse_upf(full_path)
                        break

        if val is None or nproj is None:
            missing_pseudos[sym] += 1
        else:
            total_valence += float(val)
            total_betas += int(nproj)

    # Apply total charge if present
    total_charge = qe_in.system.get("tot_charge", 0) or 0
    try:
        total_valence = float(total_valence) - float(total_charge)
    except Exception:
        pass

    if missing_pseudos:
        missing_msg = "; ".join([f"{s}:{c}" for s, c in missing_pseudos.items()])
        print(f"[WARN] Some species lacked pseudo info: {missing_msg}. "
              f"Computed total_valence={total_valence}, total_betas={total_betas}")

    # And n_el is total_valence (use int or float as you prefer)
    n_el = total_valence
    n_el_cubed = float(n_el) ** 3 
    # --- Smooth grid estimation (n_g_smooth, fft_dims) -----------------------------
    ecutwfc = qe_in.system.get("ecutwfc", None)
    ecutrho = qe_in.system.get("ecutrho", None)

    n_g_smooth = None
    fft_dims = (0, 0, 0)

    if ecutrho and ecutwfc:
        try:
            # convert structure lattice to Bohr units if needed
            ang_to_bohr = 1.889726
            lattice_bohr = np.array(structure.lattice.matrix) * ang_to_bohr

            # choose cubic or general path
            if abs(np.linalg.det(structure.lattice.matrix)) > 1e-6:
                # decide if cubic by checking approximate orthogonality and equal edges
                lengths = structure.lattice.abc
                angles = structure.lattice.angles
                if max(angles) - min(angles) < 1e-3 and max(lengths) / min(lengths) < 1.01:
                    n_g_smooth, fft_dims = compute_ng_smooth_cubic(lengths[0] * ang_to_bohr, ecutwfc, ecutrho)
                else:
                    n_g_smooth, fft_dims = compute_ng_smooth_general(lattice_bohr, ecutwfc, ecutrho)
            else:
                print("[WARN] Invalid lattice matrix, using fallback cubic estimate")
                n_g_smooth, fft_dims = compute_ng_smooth_cubic(10.0, ecutwfc, ecutrho)
        except Exception as e:
            print(f"[WARN] Failed to compute n_g_smooth: {e}")
            n_g_smooth, fft_dims = 0, (0, 0, 0)
    else:
        print("[INFO] ecutrho or ecutwfc missing -- skipping n_g_smooth computation")
        n_g_smooth, fft_dims = 0, (0, 0, 0)

    print(f"[DEBUG] n_g_smooth={n_g_smooth}, fft_dims={fft_dims}")

    # Spin
    spin = qe_in.system.get("nspin", 1)

    # nbnd estimation
    if spin == 1:
        n_states = int(np.ceil(total_valence / 2))
    else:
        n_states = int(np.ceil(total_valence))
    nbnd = int(n_states * 1.1) + 5

    # --- K-points ---
    mode, mesh, shift = parse_kpoints_from_input(input_file)
    if mode == "automatic" and mesh is not None:
        dataset = spglib.get_symmetry_dataset(
            (structure.lattice.matrix, structure.frac_coords, [site.specie.Z for site in structure])
        )
        irreducible_kpoints = len(
            spglib.get_ir_reciprocal_mesh(mesh, structure.lattice.matrix, is_shift=shift)[0]
        )
    else:
        # fallback for gamma/explicit
        irreducible_kpoints = mesh[0] if mesh else 1                 

    # Return a dict of features
    features = {
        "species_in_input": species,
        "nbnd_estimate": nbnd,  
        "total_valence": n_el,
        "n_el^3": n_el_cubed,
        "total_betas": total_betas,
        "n_atoms": len(structure),
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "irreducible_kpoints": irreducible_kpoints,
        "nspin": spin,
        "n_g_smooth": n_g_smooth,
        "fft_dims": fft_dims,      
        "pseudo_summary": pseudo_map,
        "kpoints_mode": mode,
        "kpoints_mesh": mesh,
        "kpoints_shift": shift,
        }


    return features
########################################
    
    
def flatten_for_nn(parsed, extra_inputs=None):
    """
    Flatten parsed QE input + optional extra user inputs into exact 14 features for NN.
    extra_inputs: dict for n_transition, n_lanthanid, n_g_smooth, n_cores, n_nodes, threads_per_node, n_pool
    """
    extra = extra_inputs or {}
    species = parsed.get("species_in_input", [])
    total_valence = parsed.get("total_valence")
    print(total_valence)
    nbnd = parsed.get("nbnd_estimate")
    irreducible_kpoints = parsed.get("irreducible_kpoints")
    num_betas = parsed.get("total_betas")
    ng_dmooth = parsed.get("n_g_smooth")

    return {
        "n_el": total_valence,
        "n_el^3": total_valence**3,
        "n_species": len(species),
        "n_at": len(species),  # assuming each site counts as an atom
        "n_transition": extra.get("n_transition", 0),
        "n_lanthanid": extra.get("n_lanthanid", 0),
        "n_ks": nbnd,
        "n_g_smooth": ng_dmooth, 
        "n_k": irreducible_kpoints,
        "n_betas": num_betas,
        "n_cores": extra.get("n_cores", 0),
        "n_nodes": extra.get("n_nodes", 0),
        "threads_per_node": extra.get("threads_per_node", 0),
        "n_pool": extra.get("n_pool", 0)
    }





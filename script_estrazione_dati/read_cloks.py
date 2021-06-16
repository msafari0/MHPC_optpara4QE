from collections import deque

def tot_string(s):
  return s.split('CPU')[-1].replace('WALL','').strip().replace(' ','')


def read_tot (s):
    t = s.partition('s')[0]
    part = t.partition('m')
    if part[1]=='m':
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
  with open(filename,'r') as f:
    clocklines=deque(filter(lambda s: ' CPU ' in s and ' WALL' in s, f))
  if len(clocklines) == 0:
    return None
  for prog in ['PWSCF', 'CP ']:
    totclock=deque(filter(lambda s: prog in s,clocklines),maxlen=1)
    if len(totclock) == 1:
      #res = ((prog, read_tot(totclock[0].split()[-2])),)
      res =  ((prog, read_tot(tot_string(totclock[0]))),)  
      break
  clocks = [
        (_.split()[0],float(_.split()[4].replace('s',''))) 
        for _ in clocklines
        if (not 'PWSCF' in _) and (not 'CP ' in _)]
  return tuple(clocks)+res 



def read_iterations(filename):
    with open(filename,'r') as f:
        clocklines = [_ for _ in f.readlines() if ' CPU ' in _ and ' WALL' in _]
    iterations = [
        (_.split()[0],float(_.split()[7].replace('s',''))) 
        for _ in clocklines
        if (not 'PWSCF' in _) and (not 'CP ' in _)]
    return tuple(iterations)


def read_program(filename):
  with open(filename) as f:
    startline= [_ for _ in f.readlines() if 'Program' in _ and 'starts' in _][0]
  return startline.split()[1]

def read_ndiag(line):
  ll = line.split('*')[1]
  ndiag = int(ll.split()[0])
  return ndiag*ndiag 

def read_parallel(filename):
  with open(filename,'r') as f:
    l = f.readlines()[:50]
  try:
    linetoread=[_ for _ in l if 'Number of MPI processes:' in _][0]
  except IndexError:
    return None
  res  = {'MPI tasks':int(linetoread.split()[4])}
  linetoread=[_ for _ in l if 'Threads/MPI process:' in _][0]
  res.update({'Threads': int(linetoread.split()[2])})
  try:
    linetoread=[_ for _ in l if "K-points division:     npool" in _][0]
    res.update({'npool': int(linetoread.split()[-1])})
  except IndexError:
    res.update({'npool':1})
  linetoread=[_ for _ in l if "R & G space division:" in _][0]
  r_n_g = int(linetoread.split()[-1])
  res.update({'n_RG':r_n_g})
  linetoread=[_ for _ in l if "wavefunctions fft division:" in _]
  if len(linetoread) == 0:
    wfc_fftdiv = (1,r_n_g)
  elif len(linetoread) > 0: 
    wfc_fftdiv = tuple( [ int(_) for _ in  (linetoread[0]).split()[-2:] ]) 
  res.update({"wfc_fft_division": wfc_fftdiv})
  res.update({"taskgroups": len(linetoread)==2})
  try:
    linetoread=[_ for _ in l if "distributed-memory algorithm" in _][0]
    res.update({'ndiag': read_ndiag(linetoread)})
  except IndexError:
    res.update({'ndiag':1})
  return res 

def read_gridinfo(filename, stringa):
  """
  filename: str path of the file to open
  stringa:  str string to search for selecting the line
  """
  with open(filename, 'r') as f:
    r = deque ( filter(lambda _: stringa in _, iter(f) ))[0]
  temp1,temp2 = r.split(":")[1], r.split(":")[2]
  grid_vecs = int(temp1.split()[0])
  temp2  = temp2.replace('(',' ').replace(')',' ').replace(',',' ')
  fft_dims = tuple((int(_) for _ in temp2.split()))
  return {"ngrid_vecs":grid_vecs, "fft_dims":fft_dims} 


def read_dimensions(filename):
  with open(filename,'r') as f:
    l=f.readlines()[40:60]
  s = "number of atoms/cell"
  try:
    r= [_ for _ in l if s in _][0]
  except IndexError:
    return None
  res = {'nat':int(r.split()[-1])}
  s  = "number of Kohn-Sham states="
  try:
    r = [_ for _ in l if s in _][0]
  except IndexError:
    return None
  res.update({'nbands': int(r.split()[-1])})
  s = "kinetic-energy cutoff"
  r = [_ for _ in l if s in _][0]
  res.update({'ecutwfc': float(r.split()[-2])})
  s = "charge density cutoff"
  r = [_ for _ in l if s in _][0]
  res.update({'ecutrho':float(r.split()[-2])})
  s = "unit-cell volume" 
  r = [_ for _ in l if s in _][0]
  res.update({'vol': float(r.split()[-2])})
  dense_fft_dims = read_gridinfo(filename,"Dense  grid:")  
  smooth_fft_dims = read_gridinfo(filename,"Smooth grid:")
  res.update({"Dense_grid":dense_fft_dims, "Smooth_grid":smooth_fft_dims})
  with open(filename,'r' ) as f:
    r = deque ( filter(lambda _: "Smooth grid:" in _, iter(f) ))[0]
  return res

def read_raminfo (filename):
  total_ram = read_estimated_ram(filename)
  if total_ram is None:
    return None 
  res=total_ram
  partial_ram = read_partial_ram(filename)
  if partial_ram is not None:
    res.update(partial_ram)
  return res 

def read_estimated_ram(filename):
  with open(filename, 'r') as f:
    lines = [_ for _ in filter(lambda _: "Estimated" in _ and "RAM" in _, iter(f)) ] 
  if len(lines)<3:
    return None
  temp = lines[0].split('>')[1].split()
  static = (float(temp[0]),temp[1]) 
  temp = lines[1].split('>')[1].split()
  max_dynamic = (float(temp[0]),temp[1])
  temp = lines[2].split('>')[1].split()
  total  = (float(temp[0]),temp[1])
  return {
    "static_per_process":static,
    "max_per_process":max_dynamic,
    "total":total
    }

def read_partial_ram(filename):
  with open(filename, 'r') as f:
    lines = [_ for _ in filter(lambda _:"Dynamical RAM for" in _, iter(f))]
  if len(lines) == 0:
    return None
  def read_line(_):
    temp1,temp2 = _.split(":")
    temp1 = temp1.replace("Dynamical RAM for","").strip()
    return temp1, float(temp2.split()[0]), temp2.split()[1]
  itera = ((read_line(l)[0],read_line(l)[1:]) for l in lines)
  return dict(tuple(itera)) 

def read_nkpoints(fname):
  with open(fname,'r') as fr:
    l = filter(lambda s: 'number of k points' in s, fr)
    l = deque(l,maxlen=1)
  if len(l) == 1:
    return int(l[0].split()[4])
  else:
    return None  
"""Full-catalog exact top-20 with original NumPy tie behavior preserved.

The optional C++ heap scans every item. Rows with tied selected scores, a tied
cutoff or NaNs use the original routine. No approximate candidate pruning.
"""
import ctypes,hashlib,os,subprocess,tempfile
from pathlib import Path
import numpy as np
from metrics import rank_scores as numpy_rank
_kernel=None
_attempted=False

def initialize():
 global _kernel,_attempted
 if _attempted:return
 _attempted=True
 source=Path(__file__).with_name('top20.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
 directory=Path(tempfile.gettempdir())/'cmr_exact_ranking';directory.mkdir(exist_ok=True);target=directory/(digest+'.so')
 try:
  if not target.exists():
   temporary=directory/(digest+'.'+str(os.getpid())+'.so')
   subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   temporary.replace(target)
  lib=ctypes.CDLL(str(target));kernel=lib.exact_top20
  kernel.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint8)]
  kernel.restype=None;_kernel=kernel
 except (OSError,subprocess.CalledProcessError):_kernel=None

def rank_scores(scores,X,user_ids,k=20):
 initialize();scores=np.asarray(scores)
 if _kernel is None or k!=20 or scores.dtype!=np.float32 or scores.shape[1]<20:return numpy_rank(scores,X,user_ids,k)
 seen=X[user_ids].tocoo();scores[seen.row,seen.col]=-np.inf
 contiguous=np.ascontiguousarray(scores);out=np.empty((len(scores),20),np.int32);fallback=np.empty(len(scores),np.uint8)
 _kernel(contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),len(scores),scores.shape[1],out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),fallback.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
 rows=np.flatnonzero(fallback)
 if len(rows):out[rows]=numpy_rank(scores[rows].copy(),X,np.asarray(user_ids)[rows],k)
 return out

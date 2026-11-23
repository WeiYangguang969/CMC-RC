"""Optional exact CSR x 64 dense columns, vectorized across independent users.

Each output entry follows the reference storage order. The compiler may not
contract products and sums. A numerical probe (including FMA-sensitive terms)
and feature/format checks guard the compiled path; SciPy remains the fallback.
"""
import ctypes,hashlib,os,platform,subprocess,tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

_lib=None
_kernel=None
_attempted=False
backend='SciPy reference'

def _native(b,x):
    result=np.empty((b.shape[0],64),dtype=np.float32)
    _kernel(b.shape[0],*[a.ctypes.data for a in (b.indptr,b.indices,b.data,x,result)])
    return result

def initialize():
    global _lib,_kernel,_attempted,backend
    if _attempted:return _kernel is not None
    _attempted=True
    if os.environ.get('CMR_REFERENCE_PRODUCT')=='1':return False
    try:
        flags=Path('/proc/cpuinfo').read_text().lower().split()
        if platform.machine().lower() not in ('x86_64','amd64'):return False
        isa='avx512f' if 'avx512f' in flags else ('avx2' if 'avx2' in flags else None)
        if isa is None:return False
        src=Path(__file__).with_name('ordered_product.cpp');tag=hashlib.sha256(src.read_bytes()+('v2:'+isa+':ffp-contract-off').encode()).hexdigest()[:16]
        root=Path(tempfile.gettempdir())/'cmr_ordered_product';root.mkdir(exist_ok=True);lib=root/(tag+'.so')
        if not lib.exists():
            temp=root/(tag+'.'+str(os.getpid())+'.so')
            subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-m'+isa,'-ffp-contract=off',str(src),'-o',str(temp)],check=True,capture_output=True);temp.replace(lib)
        _lib=ctypes.CDLL(str(lib));_kernel=_lib.mul64;_kernel.restype=None;_kernel.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*5
        rg=np.random.default_rng(20261027);v=rg.normal(size=(31,79)).astype('f');v[rg.random(v.shape)<.8]=0
        b=sp.csr_matrix(v);x=rg.normal(size=(79,64)).astype('f');epsilon=np.float32(2**-23)
        witness=sp.csr_matrix(np.array([[-1,1+epsilon]],dtype='f'));w=np.repeat(np.array([[1],[1-epsilon]],dtype='f'),64,axis=1)
        if any(not np.array_equal((a@y).view('u4'),_native(a,y).view('u4')) for a,y in [(b,x),(witness,w)]):_kernel=None
        if _kernel is not None:backend='ordered '+isa.upper()+' float32 x 64'
    except (OSError,ValueError,subprocess.SubprocessError,AttributeError):_kernel=None
    return _kernel is not None

def ordered_product(b,x):
    if (_kernel is not None and x.ndim==2 and x.shape==(b.shape[1],64) and x.dtype==np.float32 and x.flags.c_contiguous
        and b.format=='csr' and b.dtype==np.float32 and b.indices.dtype==np.int32 and b.indptr.dtype==np.int32
        and b.data.flags.c_contiguous and b.indices.flags.c_contiguous and b.indptr.flags.c_contiguous):return _native(b,x)
    return b@x

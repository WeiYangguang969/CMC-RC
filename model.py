"""Sparse conditional reconstruction and user/item residual filtering."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import concurrent.futures as cf
import multiprocessing as mp
import hashlib, json, time, zipfile, io
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve
from scipy.sparse.linalg import cg, LinearOperator
from sklearn.utils.extmath import randomized_svd
from utils import neighbors
from utils.runtime import peak_memory_mib
from utils.fast_rank import rank_scores, initialize as initialize_ranking
from utils.fast_product import ordered_product, initialize as initialize_product
from utils.shifted_solve import shifted_solutions

_FIT_COLUMN_FUNCTION = None
initialize_product()
initialize_ranking()


def _fit_process_column(j):
 return _FIT_COLUMN_FUNCTION(j)


def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def save_npz(path,_compress=True,**a):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 buffer=io.BytesIO();writer=np.savez_compressed if _compress else np.savez;writer(buffer,**a);payload=buffer.getbuffer()
 with zipfile.ZipFile(buffer) as z:
  bad=z.testzip()
  if bad is not None:raise IOError('NPZ serialization failure: '+bad)
 with tmp.open('wb') as f:
  for start in range(0,len(payload),2**20):
   block=payload[start:start+2**20];written=f.write(block)
   if written!=len(block):raise IOError('Short NPZ write')
  f.flush();os.fsync(f.fileno())
 if tmp.stat().st_size!=len(payload):raise IOError('Truncated NPZ output')
 with zipfile.ZipFile(tmp) as z:
  bad=z.testzip()
  if bad is not None:raise IOError('NPZ integrity failure: '+bad)
 tmp.replace(p)
 if not zipfile.is_zipfile(p):raise IOError('NPZ commit failure: '+str(p))


def save_sparse(path,B):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.stem+'.tmp.npz')
 buffer=io.BytesIO();sp.save_npz(buffer,B,compressed=False);payload=buffer.getbuffer()
 with tmp.open('wb') as f:
  for start in range(0,len(payload),2**20):
   block=payload[start:start+2**20]
   if f.write(block)!=len(block):raise IOError('Short sparse write')
  f.flush();os.fsync(f.fileno())
 with zipfile.ZipFile(tmp) as z:
  if z.testzip() is not None:raise IOError('Sparse archive integrity failure')
 tmp.replace(p)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()


def event(kind,**kw):print(json.dumps({'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**kw}),flush=True)


def base_key(c):return f"lambda{c['lambda']:g}_nu{c['nu']:g}"


def key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]


def solve_spd(M,v,start=None):
 fallback=False
 if len(v)>=192:
  inv=1/np.diag(M)
  b,info=cg(M,v,x0=start,M=LinearOperator(M.shape,matvec=lambda x:inv*x),rtol=1e-11,atol=0.,maxiter=4*len(v))
  residual=np.linalg.norm(M@b-v)/max(np.linalg.norm(v),1e-30)
  if info or residual>1e-10 or not np.isfinite(b).all():b=solve(M,v,assume_a='pos',check_finite=False);fallback=True
 else:b=solve(M,v,assume_a='pos',check_finite=False)
 residual=float(np.linalg.norm(M@b-v)/max(np.linalg.norm(v),1e-30))
 if residual>1e-9:raise ArithmeticError('Numerical solve failed: '+str(residual))
 return b,residual,fallback


def fit_bank(ds,configs,dest,workers=3):
 """Store BH and its exact positive centering scale, avoiding duplicate solves.

 All full-catalog scoring uses this same factored representation. Sparse BH and
 scaling are float32, as are the other score operators. Numerical residuals are
 checked in float64 before model storage. No conditional term is constructed for
 a standalone nu=0 call.
 """
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 unique=list({base_key(c):c for c in configs}.values())
 missing=[c for c in unique if not (dest/(base_key(c)+'.npz')).exists()]
 if not missing:return {'cached':True}
 progress=dest/'fit_progress.json';resume=json.loads(progress.read_text()) if progress.exists() else None
 if resume:
  assert set(resume['base_keys']).issubset({base_key(c) for c in unique})
  missing=resume['configs']
 if not missing:return {'cached':True}
 started=time.perf_counter();gt=time.perf_counter();G=(ds.X.T@ds.X).tocsr();G.sort_indices();gram_s=time.perf_counter()-gt
 diag=G.diagonal().astype(float);conditional=any(c['nu'] for c in missing)
 Xc=ds.X.tocsc() if conditional else None
 n=ds.n_items;k=unique[0]['k'];pool=unique[0].get('candidate_pool',k);nc=len(missing)
 assert all(c['k']==k and c.get('q',0)==0 and c.get('rho',.5)==.5 for c in unique)
 if resume:assert resume['base_keys']==[base_key(c) for c in missing]
 first=resume['completed_columns'] if resume else 0;mode='r+' if resume else 'w+'
 ixbank=np.lib.format.open_memmap(dest/'indices.partial.npy',mode=mode,dtype=np.int32,shape=(n,k))
 bank=np.lib.format.open_memmap(dest/'coefficients.partial.npy',mode=mode,dtype=np.float32,shape=(n,nc,k))
 scales=np.lib.format.open_memmap(dest/'scales.partial.npy',mode=mode,dtype=np.float32,shape=(n,nc))
 if not resume:ixbank[:]=-1;scales[:]=1
 timing={'gram_seconds':gram_s,'local_gram_cpu_seconds':0.,'conditional_cpu_seconds':0.,'solve_cpu_seconds':0.,
         'maximum_relative_residual':0.,'fallbacks':0,'resumed':resume,'conditional_constructed':conditional,
         'solver':'shared shifted Lanczos for at least 3 lambdas and width>=192; checked PCG/Cholesky otherwise'}
 order=sorted(range(nc),key=lambda i:(missing[i]['lambda'],missing[i]['nu']))
 def one(j):
  t=time.perf_counter();ix,v=neighbors(G,j,diag,pool)
  if len(ix)>k:
   active=np.sort(np.argpartition(v/np.sqrt(diag[ix]),-k)[-k:]);ix,v=ix[active],v[active]
  m=len(ix)
  if not m:return j,ix,np.empty((nc,0),np.float32),np.ones(nc,np.float32),(0.,0.,0.),0.,0
  A=G[ix][:,ix].toarray().astype(float);v=v.astype(float);local_s=time.perf_counter()-t;t=time.perf_counter()
  if conditional:
   us=Xc.indices[Xc.indptr[j]:Xc.indptr[j+1]];Z=ds.X[us][:,ix].toarray().astype(float);H=Z.T@Z
   r=max(len(us)-1,0)/(max(len(us)-1,0)+unique[0].get('support',10))
  conditional_s=time.perf_counter()-t if conditional else 0.;t=time.perf_counter()
  values=np.empty((nc,m),np.float32);scale=np.ones(nc,np.float32);warm={};maxres=0.;fallbacks=0
  shared={}
  if m>=192:
   for nu in sorted({c['nu'] for c in missing}):
    group=[i for i,c in enumerate(missing) if c['nu']==nu]
    if len(group)<3:continue
    M0=A+nu*r*H if nu else A
    try:
     solutions,details=shifted_solutions(M0,v,np.sqrt(np.maximum(diag[ix],1)),[missing[i]['lambda'] for i in group])
     shared.update({i:(b,details['maximum_relative_residual']) for i,b in zip(group,solutions)})
    except (ArithmeticError,np.linalg.LinAlgError):pass
  for ci in order:
   c=missing[ci];nu=c['nu']
   if ci in shared:b,res=shared[ci];fallback=False
   else:
    M=A.copy()
    if nu:M+=nu*r*H
    M.flat[::m+1]+=c['lambda']*np.sqrt(np.maximum(diag[ix],1))
    b,res,fallback=solve_spd(M,v,warm.get(nu))
   warm[nu]=b;maxres=max(maxres,res);fallbacks+=int(fallback)
   denominator=1-nu*r*(v@b)/diag[j] if nu else 1.
   if denominator<=0:raise ArithmeticError('Nonpositive centering scale')
   values[ci]=b;scale[ci]=1/denominator
  return j,ix,values,scale,(local_s,conditional_s,time.perf_counter()-t),maxres,fallbacks
 global _FIT_COLUMN_FUNCTION
 use_process=os.environ.get('CMR_FIT_EXECUTOR')=='process' and os.name=='posix'
 _FIT_COLUMN_FUNCTION=one if use_process else None
 executor=cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) if use_process else cf.ThreadPoolExecutor(max_workers=workers)
 column_fn=_fit_process_column if use_process else one
 timing['executor']='forked processes' if use_process else 'threads'
 with executor as ex:
  for a in range(first,n,256):
   for j,ix,values,scale,times,res,fall in ex.map(column_fn,range(a,min(a+256,n)),chunksize=16):
    ixbank[j,:len(ix)]=ix;bank[j,:,:len(ix)]=values;scales[j]=scale
    for label,val in zip(['local_gram_cpu_seconds','conditional_cpu_seconds','solve_cpu_seconds'],times):timing[label]+=val
    timing['maximum_relative_residual']=max(timing['maximum_relative_residual'],res);timing['fallbacks']+=fall
   if (a-first)%2048==0:
    ixbank.flush();bank.flush();scales.flush()
    write_json(progress,{'completed_columns':min(n,a+256),'base_keys':[base_key(c) for c in missing],
                         'prior_wall_seconds':time.perf_counter()-started,'configs':missing})
    event('fit',dataset=ds.name,columns=min(n,a+256),total=n,base_fits=nc,seconds=round(time.perf_counter()-started,1))
 _FIT_COLUMN_FUNCTION=None
 ixbank.flush();bank.flush();scales.flush()
 write_json(progress,{'completed_columns':n,'base_keys':[base_key(c) for c in missing],
                      'prior_wall_seconds':time.perf_counter()-started,'configs':missing})
 valid=ixbank>=0;indptr=np.r_[0,np.cumsum(valid.sum(1))].astype(np.int32);indices=np.asarray(ixbank[valid],np.int32)
 del G,Xc
 for ci,c in enumerate(missing):
  B=sp.csc_matrix((np.asarray(bank[:,ci,:][valid],np.float32),indices,indptr),shape=(n,n)).tocsr()
  # Model metadata and both factors are committed atomically in one NPZ.
  # Rebuildable local caches use stored NPZ members: identical numeric arrays,
  # less CPU spent compressing large banks, faster loading. User evidence
  # remains compressed and every file retains CRC/length/atomic-write checks.
  save_npz(dest/(base_key(c)+'.npz'),_compress=False,data=B.data,indices=B.indices,indptr=B.indptr,shape=np.asarray(B.shape),scale=np.asarray(scales[:,ci]))
 timing.update(wall_seconds=time.perf_counter()-started,peak_mib=peak_memory_mib(),configs=missing)
 write_json(dest/('fit_'+str(time.time_ns())+'.json'),timing)
 del ixbank,bank,scales
 for p in dest.glob('*.partial.npy'):p.unlink()
 if progress.exists():progress.unlink()
 event('fit_complete',dataset=ds.name,base_fits=nc,seconds=timing['wall_seconds'])
 return timing


def load_model(path):
 with np.load(path) as z:
  B=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']));scale=z['scale']
 return B,scale


def operators(ds,dest,workers=3,user=True,spectral=True,rank=128,seed=20261027):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);P=V=pi=None;cost={}
 if user:
  path=dest/'P.npz';weight_path=dest/'reversibility_weights.npz'
  if path.exists():P=sp.load_npz(path);pi=np.load(weight_path)['pi']
  else:
   t=time.perf_counter();X=ds.X.astype(float);Xw=X.multiply(ds.deg[None,:].astype(float)**(-.5)).tocsr();degree=np.asarray(Xw.sum(1)).ravel()
   def block(a):
    ids=np.arange(a,min(a+64,ds.n_users));A=(Xw[ids]@X.T).toarray();A/=np.sqrt(degree[None,:]);A[np.arange(len(ids)),ids]=0
    k=min(200,ds.n_users-1);ix=np.argpartition(A,-k,axis=1)[:,-k:];v=np.take_along_axis(A,ix,axis=1)
    v/=np.maximum(v.sum(1,keepdims=True),1e-30);ok=v>0
    return np.broadcast_to(ids[:,None],ix.shape)[ok].astype(np.int32),ix[ok].astype(np.int32),v[ok].astype(np.float32)
   with cf.ThreadPoolExecutor(max_workers=workers) as ex:parts=list(ex.map(block,range(0,ds.n_users,64)))
   rows,cols,vals=[np.concatenate([p[i] for p in parts]) for i in range(3)]
   K=sp.csr_matrix((vals,(rows,cols)),shape=(ds.n_users,ds.n_users));W=(K+K.T)*.5;d=np.asarray(W.sum(1)).ravel()
   P=W.multiply(1/np.maximum(d,1e-30)[:,None]).tocsr();pi=np.where(d>0,d,1.).astype(float);pi/=pi.sum()
   save_sparse(path,P);save_npz(weight_path,pi=pi);cost['user_build_seconds']=time.perf_counter()-t
 if spectral:
  path=dest/f'V{rank}_seed{seed}.npz'
  if path.exists():V=np.load(path)['V']
  else:
   t=time.perf_counter();N=ds.X.multiply(ds.hist[:,None]**(-.5)).multiply(ds.deg[None,:]**(-.5)).tocsr().astype(np.float32)
   k=min(256,min(N.shape)-1);V=randomized_svd(N,n_components=k,n_iter=7,n_oversamples=32,random_state=seed)[2][:rank]
   save_npz(path,V=V);cost['spectral_build_seconds']=time.perf_counter()-t
 write_json(dest/('access_'+str(time.time_ns())+'.json'),cost)
 return P,V,pi,cost


def score(ds,B,scale,P,V,configs,users=None,workers=3,diagnostics=False,pi=None):
 """Group H and centered-H scores using their exact column-scaling identity."""
 users=np.arange(ds.n_users) if users is None else np.asarray(users)
 need_user=any(c['gamma'] for c in configs);need_rb=any(c['gamma'] and c.get('delta',1) for c in configs)
 need_spectral=any(c['beta'] or (c['gamma'] and c.get('delta',1)!=1) for c in configs)
 BT=B.T.tocsr() if need_rb else None;features={};Q=None;t=time.perf_counter()
 for stat in {c['statistic'] for c in configs}:
  group=[c for c in configs if c['statistic']==stat];a=scale if stat=='centered' else np.ones(ds.n_items,np.float32)
  F=PF=PZ=None
  if any(c['beta'] or (c['gamma'] and c.get('delta',1)!=1) for c in group):
   T=(V*ds.deg[None,:]**(-.5)).astype(np.float32);Q=(V*ds.deg[None,:]**.5).astype(np.float32)
   Z=ds.X@(B@(T.T*a[:,None]));F=ds.X@T.T-Z
   if any(c['gamma'] and c['beta'] and c['overlap'] for c in group):PF=P@F
   if any(c['gamma'] and c.get('delta',1)!=1 for c in group):PZ=P@Z
  features[stat]=(a,F,PF,PZ)
 prep=time.perf_counter()-t;tops=[np.empty((len(users),20),np.int32) for c in configs];norms=np.zeros((len(configs),2));t=time.perf_counter()
 def block(start):
  ids=users[start:start+64];s0=(ds.X[ids]@B).toarray();raw=rb=None
  if need_user:
   px=P[ids]@ds.X;raw=px.toarray()
   if need_rb:rb=(ordered_product(BT,np.ascontiguousarray(raw.T))).T
  terms={}
  for stat,(a,F,PF,PZ) in features.items():
   group=[c for c in configs if c['statistic']==stat]
   terms[stat]=(F[ids]@Q if any(c['beta'] for c in group) else None,
                PF[ids]@Q if PF is not None else None,PZ[ids]@Q if PZ is not None else None)
  out=[];ns=np.zeros((len(configs),2));X=ds.X[ids].toarray() if diagnostics else None
  for ci,c in enumerate(configs):
   a=features[c['statistic']][0];item,cross,z=terms[c['statistic']];g,b,d=c['gamma'],c['beta'],c.get('delta',1.)
   S=s0*a[None,:]
   if g:
    S+=g*raw
    if d:S-=g*d*rb*a[None,:]
    if d!=1:S-=g*(1-d)*z
   if b:
    S+=b*item
    if g and c['overlap']:S-=g*b*cross
   if diagnostics:
    R=X.astype(float)-S;ns[ci,0]=np.sum(R*R)
    ns[ci,1]=np.sum(R*R*pi[ids,None]/ds.deg[None,:])/(1+g)**2
   out.append(rank_scores(S,ds.X,ids))
  return start,out,ns
 with cf.ThreadPoolExecutor(max_workers=workers) as ex:
  for start,result,ns in ex.map(block,range(0,len(users),64)):
   for ci,top in enumerate(result):tops[ci][start:start+len(top)]=top
   norms+=ns
 cost={'shared_preparation_seconds':prep,'shared_scoring_seconds':time.perf_counter()-t,'configurations':len(configs),'users':len(users)}
 if diagnostics:cost['norms']=[{'raw_residual_frobenius':float(x[0]**.5),'ranking_equivalent_weighted_residual_frobenius':float(x[1]**.5)} for x in norms]
 return tops,cost

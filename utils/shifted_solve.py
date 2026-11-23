"""Shared Lanczos solves for several diagonal-regularization strengths.

Diagonal whitening makes the regularization a scalar shift. Full
reorthogonalization controls the shared Krylov basis. Every returned solution
is checked against its original equation; the caller falls back on failure.
This changes numerical work, not the fitted objective or search domain.
"""
import numpy as np
from scipy.linalg import solve_banded

def shifted_solutions(A,v,diagonal,lambdas,tolerance=1e-11):
 n=len(v);lambdas=np.asarray(lambdas,float);inv=1/np.sqrt(diagonal)
 M=A*inv[:,None]*inv[None,:];rhs=v*inv;norm=float(np.linalg.norm(rhs))
 if norm==0:return np.zeros((len(lambdas),n)),{'iterations':0,'maximum_relative_residual':0.}
 Q=np.empty((n,n),dtype=float,order='F');q=rhs/norm;alpha=[];beta=[];previous=np.zeros(n);previous_beta=0.
 best=None
 for j in range(n):
  Q[:,j]=q;w=M@q-previous_beta*previous;a=float(q@w);w-=a*q
  # Twice-applied modified projection protects near-breakdown cases.
  basis=Q[:,:j+1];w-=basis@(basis.T@w);w-=basis@(basis.T@w)
  b=float(np.linalg.norm(w));alpha.append(a)
  ab=np.zeros((3,j+1));ab[1]=alpha
  if j:ab[0,1:]=beta;ab[2,:-1]=beta
  e=np.zeros(j+1);e[0]=norm;ab[1]+=lambdas.min()
  y=solve_banded((1,1),ab,e,check_finite=False)
  estimate=b*abs(y[-1])/norm
  if estimate<tolerance*.01 or b<1e-14 or j==n-1:
   values=[];residuals=[]
   for shift in lambdas:
    ab[1]=np.asarray(alpha)+shift;y=solve_banded((1,1),ab,e,check_finite=False);x=inv*(basis@y)
    residual=float(np.linalg.norm(A@x+shift*diagonal*x-v)/max(np.linalg.norm(v),1e-30));values.append(x);residuals.append(residual)
   best=(np.asarray(values),{'iterations':j+1,'maximum_relative_residual':max(residuals)})
   if max(residuals)<=tolerance:return best
  if b<1e-14:break
  beta.append(b);previous=q;previous_beta=b;q=w/b
 raise ArithmeticError('Shared shifted solves failed the original-equation residual check: '+str(None if best is None else best[1]))

#include <immintrin.h>
#include <cstdint>
extern "C" void mul64(int n,const int* __restrict ranges,const int* __restrict ix,const float* __restrict vals,const float* __restrict dense,float* __restrict out){
 for(int i=0;i<n;i++){
#if defined(__AVX512F__)
  __m512 a=_mm512_setzero_ps(),b=a,c=a,d=a;
  for(int j=ranges[i];j<ranges[i+1];j++){
   const float *row=dense+64LL*ix[j]; __m512 v=_mm512_set1_ps(vals[j]);
   a=_mm512_add_ps(a,_mm512_mul_ps(v,_mm512_loadu_ps(row)));
   b=_mm512_add_ps(b,_mm512_mul_ps(v,_mm512_loadu_ps(row+16)));
   c=_mm512_add_ps(c,_mm512_mul_ps(v,_mm512_loadu_ps(row+32)));
   d=_mm512_add_ps(d,_mm512_mul_ps(v,_mm512_loadu_ps(row+48)));
  }
  float *dest=out+64LL*i;_mm512_storeu_ps(dest,a);_mm512_storeu_ps(dest+16,b);_mm512_storeu_ps(dest+32,c);_mm512_storeu_ps(dest+48,d);
#else
  __m256 acc[8];
  for(int k=0;k<8;k++) acc[k]=_mm256_setzero_ps();
  for(int j=ranges[i];j<ranges[i+1];j++){
   const float *row=dense+64LL*ix[j];const __m256 v=_mm256_set1_ps(vals[j]);
   for(int k=0;k<8;k++) acc[k]=_mm256_add_ps(acc[k],_mm256_mul_ps(v,_mm256_loadu_ps(row+8*k)));
  }
  for(int k=0;k<8;k++) _mm256_storeu_ps(out+64LL*i+8*k,acc[k]);
#endif
 }
}

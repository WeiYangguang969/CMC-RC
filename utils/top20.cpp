#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
using HeapEntry=std::pair<float,int32_t>;
__attribute__((target("avx512f")))
static void scan_vector(const float* x,int64_t columns,HeapEntry* heap,bool& nan){
 int64_t j=20;
 for(;j+16<=columns;j+=16){
  __m512 v=_mm512_loadu_ps(x+j);
  nan|=bool(_mm512_cmp_ps_mask(v,v,_CMP_UNORD_Q));
  unsigned mask=_mm512_cmp_ps_mask(v,_mm512_set1_ps(heap[0].first),_CMP_GT_OQ);
  while(mask){int k=__builtin_ctz(mask);mask&=mask-1;float value=x[j+k];
   if(value>heap[0].first){std::pop_heap(heap,heap+20,std::greater<HeapEntry>());heap[19]={value,static_cast<int32_t>(j+k)};std::push_heap(heap,heap+20,std::greater<HeapEntry>());}
  }
 }
 for(;j<columns;++j){float value=x[j];nan|=std::isnan(value);if(value>heap[0].first){std::pop_heap(heap,heap+20,std::greater<HeapEntry>());heap[19]={value,static_cast<int32_t>(j)};std::push_heap(heap,heap+20,std::greater<HeapEntry>());}}
}
__attribute__((target("avx512f")))
static bool tied_cutoff_vector(const float* x,int64_t columns,float cutoff){
 int count=0;int64_t j=0;__m512 c=_mm512_set1_ps(cutoff);
 for(;j+16<=columns&&count<2;j+=16)count+=__builtin_popcount(static_cast<unsigned>(_mm512_cmp_ps_mask(_mm512_loadu_ps(x+j),c,_CMP_EQ_OQ)));
 for(;j<columns&&count<2;++j)count+=x[j]==cutoff;
 return count>1;
}
#endif
extern "C" void exact_top20(const float* input, int64_t rows, int64_t columns, int32_t* output, uint8_t* fallback) {
 using Entry=std::pair<float,int32_t>;
#if defined(__x86_64__) && defined(__GNUC__)
 const bool vectorized=__builtin_cpu_supports("avx512f");
#else
 const bool vectorized=false;
#endif
 for(int64_t r=0;r<rows;++r){
  const float* x=input+r*columns;Entry heap[20];bool nan=false;
  for(int j=0;j<20;++j){bool bad=std::isnan(x[j]);nan|=bad;heap[j]={bad?-INFINITY:x[j],j};}
  std::make_heap(heap,heap+20,std::greater<Entry>());
#if defined(__x86_64__) && defined(__GNUC__)
  if(vectorized)scan_vector(x,columns,heap,nan);else
#endif
  for(int64_t j=20;j<columns;++j){float v=x[j];nan|=std::isnan(v);if(v>heap[0].first){std::pop_heap(heap,heap+20,std::greater<Entry>());heap[19]={v,static_cast<int32_t>(j)};std::push_heap(heap,heap+20,std::greater<Entry>());}}
  std::sort(heap,heap+20,std::greater<Entry>());bool ties=false;
  for(int j=1;j<20;++j)ties|=heap[j].first==heap[j-1].first;
  if(!ties&&!nan){float cutoff=heap[19].first;
#if defined(__x86_64__) && defined(__GNUC__)
   if(vectorized)ties=tied_cutoff_vector(x,columns,cutoff);else
#endif
   {int count=0;for(int64_t j=0;j<columns&&count<2;++j)count+=x[j]==cutoff;ties=count>1;}}
  fallback[r]=nan||ties;for(int j=0;j<20;++j)output[r*20+j]=heap[j].second;
 }
}

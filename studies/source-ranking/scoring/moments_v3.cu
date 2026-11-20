// Isolated research scorer. Buffer layout follows Inria diff-gaussian-rasterization.
// See copied LICENSE.md. Does not change the installed renderer.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
template<class T> static void obtain(char*& chunk,T*& ptr,size_t count){
 uintptr_t off=(reinterpret_cast<uintptr_t>(chunk)+127)&~uintptr_t(127);
 ptr=reinterpret_cast<T*>(off);chunk=reinterpret_cast<char*>(ptr+count);
}
struct Buffers {float2* xy;float4* conic;float* rgb;uint32_t* ids;uint2* ranges;};
Buffers unpack(torch::Tensor geom,torch::Tensor bin,torch::Tensor img,int N,int W,int H){
 Buffers b;char* q=reinterpret_cast<char*>(geom.data_ptr<uint8_t>());
 float* depths;bool* clamp;int* radii;float* cov;
 obtain(q,depths,N);obtain(q,clamp,3*N);obtain(q,radii,N);obtain(q,b.xy,N);
 obtain(q,cov,6*N);obtain(q,b.conic,N);obtain(q,b.rgb,3*N);
 q=reinterpret_cast<char*>(bin.data_ptr<uint8_t>());obtain(q,b.ids,0);
 q=reinterpret_cast<char*>(img.data_ptr<uint8_t>());float* ta;uint32_t* nc;
 obtain(q,ta,W*H);obtain(q,nc,W*H);obtain(q,b.ranges,W*H);return b;
}
__device__ float alpha_at(int id,int x,int y,const float2* xy,const float4* conic){
 float2 u=xy[id];float4 c=conic[id];float dx=u.x-x,dy=u.y-y;
 float power=-.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
 if(power>0)return 0;
 float a=fminf(.99f,c.w*expf(power));return a<1.f/255.f?0:a;
}
__global__ void score_kernel(Buffers b,int N,int W,int H,float p,const float* bg,float* score,float* ideal,bool mean_only){
 int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
 if(x>=W||y>=H)return;int pix=y*W+x;
 uint2 range=b.ranges[blockIdx.y*((W+15)/16)+blockIdx.x];
 double logz=0,logt=0;float T=1,C[3]={0,0,0};
 for(uint32_t k=range.x;k<range.y;k++){
  int id=b.ids[k];float a=alpha_at(id,x,y,b.xy,b.conic);if(a==0)continue;
  double da=a,dp=p,z=(1.-dp)+dp*(1.-da)*(1.-da);
  if(!mean_only)logz+=(p==1.f?2.0*log1pf(-a):logf((float)z));logt+=log1pf(-p*a);
  for(int c=0;c<3;c++)C[c]+=T*a*b.rgb[3*id+c];
  T*=1-a;
 }
 for(int c=0;c<3;c++)ideal[c*H*W+pix]=C[c]+T*bg[c];
 float mu[3]={bg[0],bg[1],bg[2]},var=0;
 float norm=1.f/(3.f*W*H);
 for(int k=(int)range.y-1;k>=(int)range.x;k--){
  int id=b.ids[k];float a=alpha_at(id,x,y,b.xy,b.conic);if(a==0)continue;
  double da=a,dp=p,z=(1.-dp)+dp*(1.-da)*(1.-da);
  if(!mean_only)logz-=(p==1.f?2.0*log1pf(-a):logf((float)z));logt-=log1pf(-p*a);
  float dist=0;for(int c=0;c<3;c++){float d=b.rgb[3*id+c]-mu[c];dist+=d*d;}
  float u=mean_only?0.f:expf((float)logz),tt=expf((float)(2*logt));
  atomicAdd(score+id,mean_only?norm*a*a*tt*dist:norm*a*a*u*(dist+var));
  atomicAdd(score+N+id,norm*a*a*tt*dist);
  if(!mean_only)var=(float)z*var+p*(1-p)*a*a*dist;
  for(int c=0;c<3;c++)mu[c]=p*a*b.rgb[3*id+c]+(1-p*a)*mu[c];
 }
}
std::vector<torch::Tensor> moment_score(torch::Tensor geom,torch::Tensor bin,torch::Tensor img,torch::Tensor bg,int64_t N,int64_t W,int64_t H,double p){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda()&&bg.is_cuda(),"CUDA tensors required");
 TORCH_CHECK(p>0&&p<=1,"p in (0,1]");
 auto scores=torch::zeros({2,N},bg.options());auto ideal=torch::empty({3,H,W},bg.options());
 Buffers b=unpack(geom,bin,img,N,W,H);
 score_kernel<<<dim3((W+15)/16,(H+15)/16),dim3(16,16),0,c10::cuda::getCurrentCUDAStream()>>>(b,N,W,H,(float)p,bg.data_ptr<float>(),scores.data_ptr<float>(),ideal.data_ptr<float>(),false);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {scores,ideal};
}
std::vector<torch::Tensor> moment_score_mean(torch::Tensor geom,torch::Tensor bin,torch::Tensor img,torch::Tensor bg,int64_t N,int64_t W,int64_t H,double p){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda()&&bg.is_cuda(),"CUDA tensors required");
 TORCH_CHECK(p>0&&p<=1,"p in (0,1]");
 auto scores=torch::zeros({2,N},bg.options());auto ideal=torch::empty({3,H,W},bg.options());
 Buffers b=unpack(geom,bin,img,N,W,H);
 score_kernel<<<dim3((W+15)/16,(H+15)/16),dim3(16,16),0,c10::cuda::getCurrentCUDAStream()>>>(b,N,W,H,(float)p,bg.data_ptr<float>(),scores.data_ptr<float>(),ideal.data_ptr<float>(),true);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {scores,ideal};
}
__global__ void extract_kernel(Buffers b,int W,int H,const int64_t* pixels,int L,int cap,int64_t* ids,float* alphas,float* colors,int* counts){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=L)return;
 int pix=pixels[j],x=pix%W,y=pix/W;uint2 range=b.ranges[(y/16)*((W+15)/16)+(x/16)];int count=0;
 for(uint32_t k=range.x;k<range.y;k++){
  int id=b.ids[k];float a=alpha_at(id,x,y,b.xy,b.conic);if(a==0)continue;
  if(count<cap){ids[j*cap+count]=id;alphas[j*cap+count]=a;for(int c=0;c<3;c++)colors[(j*cap+count)*3+c]=b.rgb[id*3+c];}
  count++;
 }counts[j]=count;
}
std::vector<torch::Tensor> extract_rays(torch::Tensor geom,torch::Tensor bin,torch::Tensor img,torch::Tensor bg,torch::Tensor pixels,int64_t N,int64_t W,int64_t H,int64_t cap){
 auto L=pixels.numel();auto ids=torch::full({L,cap},-1,pixels.options());auto a=torch::zeros({L,cap},bg.options());auto c=torch::zeros({L,cap,3},bg.options());auto counts=torch::zeros({L},pixels.options().dtype(torch::kInt32));Buffers b=unpack(geom,bin,img,N,W,H);
 extract_kernel<<<(L+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(b,W,H,pixels.data_ptr<int64_t>(),L,cap,ids.data_ptr<int64_t>(),a.data_ptr<float>(),c.data_ptr<float>(),counts.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,a,c,counts};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("score",&moment_score);m.def("score_mean",&moment_score_mean);m.def("extract",&extract_rays);}

#include <error.h>
#include <gpu_types.h>
#include <blocksparse_hgemm_nc_op_gpu.cu>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define OP_N 0
#define OP_T 1

static void ClosestDivisorTo4(uint val, bool isA, uint* div, uint* res) {
  if ((val % 4) == 0) {
    *div = 4;
    *res = val / 4;
  } else if ((val % 3) == 0) {
    *div = 3;
    *res = val / 3;
  } else if ((val % 5) == 0) {
    *div = 5;
    *res = val / 5;
  } else if ((val % 2) == 0) {
    *div = 2;
    *res = val / 2;
  } else if ((val % 7) == 0) {
    *div = 7;
    *res = val / 7;
  } else if (isA) {
    *div = val;
    *res = 1;
  } else {
    *div = 1;
    *res = val;
  }
}

// From what I can tell, only ehalf is implemented.
cudaError_t hgemm_blocksparse_nx_dsd(const ehalf* X,
                                     const ehalf* W,
                                     ehalf* Y,
                                     bsmm_params* params,
                                     uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const bhalf* X,
                                     const bhalf* W,
                                     bhalf* Y,
                                     bsmm_params* params,
                                     uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const float* X,
                                     const float* W,
                                     float* Y,
                                     bsmm_params* params,
                                     uint op);

// See matmul.py.
template <typename in_value_t, typename out_value_t, typename idx_t>
torch::Tensor openai_blocked_spmm(const torch::Tensor& input,
                                  const torch::Tensor& weight,
                                  const torch::Tensor& bias,
                                  uint op) {
  // SETUP (ignore)
  int SMs;
  if (major_ == 0) {
    SMs = GetCountSMsVersion(&major_, NULL);
    error::throw_if_exception(major_ < 7, "Tensorcore GPU required");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bsmm_params params;
  params.stream = stream;
  params.Lock = nullptr;

  // Notes from python:
  //     def i_shape(self, N): return (N, self.C) if self.axis else (self.C, N)
  // def o_shape(self, N): return (N, self.K) if self.axis else (self.K, N)
  // self.C  = CB * block_size
  // self.K  = KB * block_size

  // My notes:
  // input == X == A
  // weight == W == B
  // output == Y == C

  // N == N
  // K == M
  // C == K

  //   X   *   W   =   Y
  //   A   *   B   =   C
  // N X K * K X M = N X M

  bool tensorcores = major_ >= 7 && std::is_same<T1, ehalf>::value;

  int N = input.size(0);  // batch_size
  int K = input.size(1);  // in_features
  int M = bias.size(0);   // out_features

  // Idk if this is correct;
  params.N = N;
  params.K = M;
  params.C = K;

  params.bsize = 32;
  params.beta = 1.0f;
  params.pcount = 0;     // ???
  params.shared = 0;     // ???
  params.Lut = nullptr;  // ??? Look up table to access sparse matrix?
  params.Lock = 0;       // ???
  params.segments = 0;   // ??? Number of blocks?
  params.Gate = 0;       // ??? Works fine if you set to 0.

  auto options = torch::TensorOptions().device(torch::kCUDA);
  torch::Tensor output = torch::empty({N, M}, options);
  output = bias.view({M, 1}).expand({N, M}).contiguous();

  int rankA = X.ndim();
  int blkN = 128, gridN = CEIL_DIV(N, 128), modN128 = N & 127;
  if (!tensorcores || (modN128 > 0 && modN128 <= 64) ||
      gridN * params.segments < SMs * 4) {
    blkN = 64;
    gridN = CEIL_DIV(N, 64);
  }

  if (params.blk_A == 0) {
    ClosestDivisorTo4(params.segments, true, &params.blk_a, &params.blk_A);
    ClosestDivisorTo4(gridN, false, &params.blk_b, &params.blk_B);
  }

  const in_value_t* pA = input.data_ptr<in_value_t>();
  const in_value_t* pB = weight.data_ptr<in_value_t>();
  out_value_t* pC = output.data_ptr<out_value_t>();

  cudaError_t res = hgemm_blocksparse_nx_dsd(pA, pB, pC, &params, OP_N);
  error::throw_if_exception(res);

  return output;
}
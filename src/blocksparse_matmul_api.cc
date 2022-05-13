#include <iostream>
#include "gpu_types.h"
#include "error.h"

static void ClosestDivisorTo4(uint val, bool isA, uint* div, uint* res)
{
         if ((val % 4) == 0) { *div = 4; *res = val / 4; }
    else if ((val % 3) == 0) { *div = 3; *res = val / 3; }
    else if ((val % 5) == 0) { *div = 5; *res = val / 5; }
    else if ((val % 2) == 0) { *div = 2; *res = val / 2; }
    else if ((val % 7) == 0) { *div = 7; *res = val / 7; }
    else if (isA) { *div = val; *res =   1; }
    else          { *div = 1;   *res = val; }
}

#define FPROP_OP 0
#define BPROP_OP 1
#define UPDAT_OP 2

#define OP_N 0
#define OP_T 1

template <bool Fprop, CTYPE(T)>
cudaError_t BsmmXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params);

template <CTYPE(T)>
cudaError_t BsmmUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params);

// template <bool Fprop, CTYPE(T)>
// cudaError_t BsmmGatedXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params);

// template <CTYPE(T)>
// cudaError_t BsmmGatedUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_xn_64_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_64_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_64_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nt_64_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_64_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_64_dds(const float* X, const float* E, float* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_xn_128_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_128_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_128_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nt_128_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_128_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_128_dds(const float* X, const float* E, float* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_nx_dsd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_tn_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_tn_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_tn_dds(const float* X, const float* E, float* U, bsmm_params* params);

template <uint OP, MTYPE(T)>
class BlocksparseMatmulOp /* : public OpKernel */
{
public:
    // /* explicit */ BlocksparseMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), major_(0), repeat_(1), flops_(0.0f)

    BlocksparseMatmulOp(bsmm_params& parameters) : params_(parameters), SMs_(0), major_(0), repeat_(1), flops_(0.0f), gated_dw(false)
    {
        // Note this is loading "params" with the passed context variables.
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("segments", &params_.segments)/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("locks",    &params_.locks   )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("blocks",   &params_.blocks  )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("bsize",    &params_.bsize   )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("C",        &params_.C       )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("K",        &params_.K       )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("shared",   &params_.shared  )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("alpha",    &params_.alpha   )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("beta",     &params_.beta    )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("gated_dw", &gated_dw_       )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("axis",     &axis_           )/* ) */;
        // /* OP_REQUIRES_OK(ctx,  */ctx->GetAttr("bench",    &bench_          )/* ) */;

        params_.pcount = 1;
        params_.blk_A  = 0;

        error::throw_if_exception(params_.K >= params_.bsize*65536, "K < bsize*65536");
        error::throw_if_exception(params_.C >= params_.bsize*65536, "C < bsize*65536");

    }

    void Compute(bsmm_params& parameters)
    {
        if (major_ == 0)
        {
            SMs_ = GetCountSMsVersion(&major_, NULL);
            error::throw_if_exception(major_ < 7, "Tensorcore GPU required");
        }
        
        this->Compute_Xprop(parameters, OP);
    }

    void Compute_Xprop(bsmm_params& parameters, uint op)
    {
        const Tensor& A = ctx->input(0);
        const Tensor& B = ctx->input(1);
        const Tensor& L = ctx->input(2);

        OpInputList gate;
        ctx->input_list("gate", &gate);

        TensorShape shapeC;
        int N     = 1;
        int rankA = A.dims();
        for (int i = 0; i < rankA; i++)
            if (i != axis_)
            {
                shapeC.AddDim(A.dim_size(i));
                N *= A.dim_size(i);
            }
            else
                shapeC.AddDim(params_.K);

        bool tensorcores = major_ >= 7 && std::is_same<T1, ehalf>::value;

        int blkN = 128, gridN = CEIL_DIV(N, 128), modN128 = N & 127;
        if (!tensorcores || axis_ == 1 || (modN128 > 0 && modN128 <= 64) || gridN * params_.segments < SMs_*4)
        {
            blkN  = 64;
            gridN = CEIL_DIV(N, 64);
        }

        Tensor* C;
        Status s = ctx->allocate_output(0, shapeC, &C);
        if (!s.ok()) return s;

        Tensor* Lock;
        TensorShape shapeL;
        if (params_.locks > 0)
            shapeL.AddDim(gridN * params_.locks * 2);
        s = ctx->allocate_output(1, shapeL, &Lock);
        if (!s.ok()) return s;

        params_.Lock = params_.locks > 0 ? Lock->flat<int32>().data() : nullptr;
        params_.N    = N;
        params_.Lut  = (const int*)L.flat<int64>().data();
        // params_.Gate = gate.size() > 0 ? gate[0].flat<float>().data() : NULL;

        if (params_.blk_A == 0)
        {
            ClosestDivisorTo4(params_.segments, true, &params_.blk_a, &params_.blk_A);
            ClosestDivisorTo4(gridN,           false, &params_.blk_b, &params_.blk_B);
        }

        // This is where the actual flat view is being used.
        // IMPORTANT!
        const T1* pA = (const T1*)A.flat<T>().data();
        const T1* pB = (const T1*)B.flat<T>().data();
              T1* pC = (      T1*)C->flat<T>().data();

        cudaError_t res;
        for (int r = 0; r < repeat_; r++)
            if (tensorcores)
            {
                if (axis_ == 0)
                    if (blkN == 64)
                        res = hgemm_blocksparse_xn_64_sdd( pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                    else
                        res = hgemm_blocksparse_xn_128_sdd(pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                else
                    res = hgemm_blocksparse_nx_dsd(pA, pB, pC, &params_, op == FPROP_OP ? OP_N : OP_T);
            }
            else
            {
                if (params_.Gate == NULL && axis_ == 0)
                {
                    res = BsmmXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                }
                else
                {
                    // Cuda update for Volta broke these kernels.  Need to fix.
                    // Ideally merge gated and non-gated code like is done with hgemm kernels.
                    error::throw_if_exception(true, "Gated blocksparse matmul currently only supported on fp16 tensorcores.");
                    // if (op == NN_OP)
                    //     res = BsmmGatedXprop_CN<false,NTYPE(T)>(pA, pB, pC, &params_);
                    // else
                    //     res = BsmmGatedXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                }
            }
        error::throw_if_exception(res);
    }
    
    bsmm_params params_;
    int   axis_,  repeat_, SMs_, major_, grid_n_;
    float flops_;
    bool  gated_dw_;
};

Status XpropShape(InferenceContext* ctx)
{
    int    K; TF_RETURN_IF_ERROR(ctx->GetAttr(   "K",    &K));
    int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

    // C ==> K
    ShapeHandle x = ctx->input(0);
    int rank = ctx->Rank(x);
    //printf("XpropShape: %d\n", rank);
    if (rank > 0)
    {
        std::vector<DimensionHandle> shape;
        shape.reserve(rank);
        for (int i = 0; i < rank; i++)
            shape.push_back(i == axis ? ctx->MakeDim(K) : ctx->Dim(x, i));

        ctx->set_output(0, ctx->MakeShape(shape));
    }
    else
        ctx->set_output(0, ctx->UnknownShape());
    ctx->set_output(1, ctx->UnknownShape());
    return Status::OK();
}

REGISTER_OP("BlocksparseMatmul")
    // Inputs.
    .Input("x: T")
    .Input("w: T")
    .Input("lut: int64")
    .Input("lut_dx: int64")
    .Input("lut_dw: int64")
    .Input("gate: ngate * float")

    // Outputs.
    .Output("y: T")
    .Output("temp: int32")

    // Attributes are like CLI parameters.
    .Attr("T: {half, float, bfloat16}")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("segments_dx: int = 0")
    .Attr("locks: int = 0")
    .Attr("locks_dx: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("shared_dx: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    // .Attr("gated_dw: bool = false")
    // .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    // .Attr("ngate: int >= 0")
    .SetShapeFn(XpropShape)
    
    // Documentation.
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

// Registered for floats.
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),BlocksparseMatmulOp<FPROP_OP, FLOAT_V>);

// Registered for halfs.
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BlocksparseMatmulOp<FPROP_OP, EHALF_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BlocksparseMatmulOp<FPROP_OP, BHALF_V>);
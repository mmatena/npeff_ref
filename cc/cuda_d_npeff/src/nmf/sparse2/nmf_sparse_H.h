#pragma once

#include <ctime>
#include <chrono>
#include <vector>


// Debugging import
#include <cuda/debugging/debugging.h>


#include <cuda/cuda_context.h>

#include <cuda/host/dense_matrix.h>
#include <cuda/host/sparse_matrix.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>

#include <cuda/transfers.h>
#include <cuda/ops/matmul.h>
#include <cuda/ops/transpose.h>
#include <cuda/ops/matrix_conversion.h>
#include <cuda/ops/nccl_ops.h>
#include <cuda/ops/random.h>
#include <cuda/ops/frobenius_product.h>
#include <cuda/ops/spsp_matmul.h>
#include <cuda/ops/sp_slicing.h>

#include <nmf/kernels/mu.h>


#include "./only_W_learner_params.h"


namespace Nmf {
namespace OnlyW {
namespace SparseH {


template<typename InIndT>
class Learner {
public:
    using IndT = int32_t;

    // TODO: Maybe make these accessable via Learner::[name]. IDK if this works
    // or if I need to use something like typedef.
    using Host_CsrMatrix = Cuda::Host::CsrMatrix<InIndT>;
    using Device_CsrMatrix_In = Cuda::Device::CsrMatrix<InIndT>;
    using Device_CsrMatrix = Cuda::Device::CsrMatrix<IndT>;

    using Host_DenseMatrix = Cuda::Host::DenseMatrix;
    using Device_DenseMatrix = Cuda::Device::DenseMatrix;


    template<MatrixOrder order = COL_MAJOR>
    using DnMatrix = Cuda::Device::DnMatrix<order>;

    using Params = OnlyW::LearnerParams<IndT, Host_CsrMatrix, Host_CsrMatrix>;
    using DcParams = OnlyW::LearnerDeviceContextParams<IndT, Host_CsrMatrix, Host_CsrMatrix, Params>;
 
    using FrobeniousInnerProduct = Cuda::Ops::FrobeniousInnerProduct;
    using DenseDenseMatMul = Cuda::Ops::DenseDenseMatMul;

    using CsrTranspose = Cuda::Ops::CsrTranspose<IndT>;
    using SpSpMatmul = Cuda::Ops::Partitioned_SpSpMatmul_ToDense_SingleUse<IndT>;

    using SplitByCols = Cuda::Ops::SplitByCols<InIndT, IndT>;
    using SplitInfo = Cuda::Ops::SplitInfo;
    using SplitUniformity = Cuda::Ops::SplitUniformity;

    using ReIndexWithInt32 = Cuda::Ops::ReIndexWithInt32<InIndT>;

    template<MatrixOrder order = COL_MAJOR>
    using CsrToDense = Cuda::Ops::CsrToDense<IndT, order>;

    template<typename DenseT>
    using DenseAllReduce = Cuda::Ops::DenseAllReduce<DenseT>;


    using MultiplicativeUpdate = Nmf::Ops::MultiplicativeUpdate;


protected:
    Params& params;

    HostCudaContext hctx;
    DeviceCudaContext& dctx_main;

    int n_partitions;

    // std::vector<Device_CsrMatrix> H_partitions;
    // std::vector<Device_CsrMatrix> A_partitions;

    Device_DenseMatrix HH;
    Device_DenseMatrix AH;

    Device_DenseMatrix W;
    Device_DenseMatrix WHH;

    float squared_norm_A = 0.0f;


public:
    Learner(Params& params) :
        params(params),
        n_partitions(params.n_partitions),
        hctx(HostCudaContext(1, params.seed)),
        dctx_main(hctx.device_contexts[0]),
        HH(params.n_components(), params.n_components()),
        AH(params.n_examples(), params.n_components()),
        W(params.n_examples(), params.n_components()),
        WHH(params.n_examples(), params.n_components())
    {}

    void Run() {
        FirstInitializations();
        hctx.SynchronizeStreams();

        RunFirstStage();
        hctx.SynchronizeStreams();
        std::cout << "Finished running first stage.\n";

        RunSecondStage();
        hctx.SynchronizeStreams();
        std::cout << "Finished running second stage.\n";
    }

    Host_DenseMatrix GetOnHost_W() {
        Host_DenseMatrix host_W(W.n_rows, W.n_cols);
        dctx_main.CopyToHostAsync(host_W, W);
        dctx_main.SynchronizeStream();
        return host_W;
    }

protected:
    /////////////////////////////////////////////////////////////////////////////////////
    // Some initialization stuff.
    // 
    // Not all initialization happens here, but the allocations whose memory requirements
    // are known should happen here.


    void FirstInitializations() {
        W.AllocMemory(dctx_main);
        WHH.AllocMemory(dctx_main);

        HH.AllocMemory(dctx_main);
        AH.AllocMemory(dctx_main);
    }


    /////////////////////////////////////////////////////////////////////////////////////
    // First stage stuff: Computing the AH and HH dense matrices.

    // TODO: Allow computation of nnzs per col on the CPU so we can split columns by that.
    void RunFirstStage() {
        // First do everything that involves only H.

        std::vector<Device_CsrMatrix> H_partitions = PutSplitsOnDevice(*params.H);
        hctx.SynchronizeStreams();
        std::cout << "H placed on device.\n";

        std::vector<Device_CsrMatrix> HT_partitions = CreateHT(H_partitions);
        hctx.SynchronizeStreams();
        std::cout << "HT created.\n";

        ComputeProductWithHT(H_partitions, HT_partitions, HH);
        FreeDeviceAllocs_Vector(H_partitions);
        std::cout << "HH computed.\n";
        hctx.SynchronizeStreams();

        // Now do the stuff involving A.

        ComputeAH(HT_partitions);
        hctx.SynchronizeStreams();

        FreeDeviceAllocs_Vector(HT_partitions);
        hctx.SynchronizeStreams();
    }

    // Also computes the norm of A.
    void ComputeAH(std::vector<Device_CsrMatrix>& HT_partitions) {
        auto& A_row_wise_partitions = params.A_row_wise_partitions;

        long row_offset = 0;
        for(long i=0;i<A_row_wise_partitions.size();i++) {
            auto& A_rows_chunk = A_row_wise_partitions[i];

            // TODO: Allocate the memory for the largest chunk once and keep re-using it.
            Device_DenseMatrix AH_rows_chunk(A_rows_chunk.n_rows, params.n_components());

            // TODO: Remove when reusing memory for this.
            AH_rows_chunk.AllocMemory(dctx_main);

            // The A_rows_chunk split by columns.
            std::vector<Device_CsrMatrix> A_chunks = PutSplitsOnDevice(A_rows_chunk);
            ComputeAndAccumulate_SquaredNormA(A_chunks);
            hctx.SynchronizeStreams();

            ComputeProductWithHT(A_chunks, HT_partitions, AH_rows_chunk);
            FreeDeviceAllocs_Vector(A_chunks);

            MoveRowsChunkIntoAH(AH_rows_chunk, row_offset);

            // TODO: Remove when reusing memory for this.
            dctx_main.FreeDeviceAllocs(AH_rows_chunk);

            row_offset += A_rows_chunk.n_rows;
        }
    }


    void MoveRowsChunkIntoAH(Device_DenseMatrix& chunk, long row_offset) {
        THROWSERT(chunk.n_cols == AH.n_cols);
        for(long i=0;i<chunk.n_cols;i++) {
            dctx_main.CopyOnDeviceAsync(
                AH.data + i * AH.n_rows + row_offset,
                chunk.data + i * chunk.n_rows,
                chunk.n_rows
            );
        }
        dctx_main.SynchronizeStream();
    }

    // TODO: The role of split_infos is pretty ugly here [in the old version that had it]. Make cleaner.
    std::vector<Device_CsrMatrix> PutSplitsOnDevice(Host_CsrMatrix& M) {
        dctx_main.SetDevice();

        // Move the matrix to the device.
        Device_CsrMatrix_In dev_M(M.n_rows, M.n_cols, M.nnz);
        dev_M.AllocMemory(dctx_main);
        dctx_main.CopyToDeviceAsync(dev_M, M);
        // std::cout << "Copied to device.\n";

        // Perform the splitting operation.
        SplitByCols split_op(dctx_main, dev_M, n_partitions, SplitUniformity::ROWS);
        std::vector<Device_CsrMatrix> splits = split_op.Call();
       
        // Free the memory associated with the original device matrix.
        dctx_main.FreeDeviceAllocs(dev_M);

        return splits;
    }


    void ComputeAndAccumulate_SquaredNormA(std::vector<Device_CsrMatrix>& A_partitions) {
        for(auto& A : A_partitions) {
            Device_DenseMatrix values = A.ViewValuesAsVector();

            FrobeniousInnerProduct fip_op(dctx_main, values, values);
            fip_op.SetUpAsync();
            fip_op.CallAsync();

            dctx_main.SynchronizeStream();
            this->squared_norm_A += fip_op.Result();

            dctx_main.FreeDeviceAllocs(fip_op);
        }
    }

    void ComputeProductWithHT(
        std::vector<Device_CsrMatrix>& M_partitions,
        std::vector<Device_CsrMatrix>& HT_partitions,
        Device_DenseMatrix& out
    ) {
        SpSpMatmul op(dctx_main, M_partitions, HT_partitions, out);
        op.SetUp();
        op.Call();
        dctx_main.FreeDeviceAllocs(op);
    }

    std::vector<Device_CsrMatrix> CreateHT(std::vector<Device_CsrMatrix>& H_partitions) {
        std::vector<Device_CsrMatrix> HT_partitions;
        for(int i=0;i<n_partitions;i++) {
            auto& h = H_partitions[i];
            HT_partitions.emplace_back(h.n_cols, h.n_rows, h.nnz);
            HT_partitions[i].AllocMemory(dctx_main);
        }

        std::vector<CsrTranspose> ops;
        for(int i=0;i<n_partitions;i++) {
            auto& h = H_partitions[i];
            auto& ht = HT_partitions[i];
            ops.emplace_back(dctx_main, h, ht);
        }

        CsrTranspose::Perform_SingleUse(ops);
        hctx.SynchronizeStreams();

        return HT_partitions;
    }


    /////////////////////////////////////////////////////////////////////////////////////
    // Second stage stuff: The multiplicative update steps.

    void RunSecondStage() {
        int log_every_n_steps = params.log_every_n_steps;

        // Initialize W.
        Cuda::Ops::RandomGenerator rand_gen(dctx_main);
        rand_gen.InitializeUniformAsync(W);

        // Create the op for the matrix multiply.
        DenseDenseMatMul WHH_op(dctx_main, W, HH, WHH);
        MultiplicativeUpdate mu_op(dctx_main, W, AH, WHH, params.eps);

        // Stuff for computing the loss.
        FrobeniousInnerProduct neg_fip_op(dctx_main, AH, W);
        FrobeniousInnerProduct pos_fip_op(dctx_main, WHH, W);

        neg_fip_op.SetUpAsync();
        pos_fip_op.SetUpAsync();
        hctx.SynchronizeStreams();
        
        auto t_start = std::chrono::high_resolution_clock::now();
        for(long i=0;i<params.maxIters;i++) {
            DoMuStep(WHH_op, mu_op);
            
            if(log_every_n_steps > 0 && ((i + 1) % log_every_n_steps) == 0) {
                hctx.SynchronizeStreams();

                auto t_end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

                neg_fip_op.CallAsync();
                pos_fip_op.CallAsync();
                hctx.SynchronizeStreams();

                float loss = sqrt(squared_norm_A + pos_fip_op.Result() - 2.0f * neg_fip_op.Result());

                std::cout << "step " << i + 1 << ": " << loss << " [" << elapsed_ms / (double) log_every_n_steps << " ms/step]\n";
                
                t_start = std::chrono::high_resolution_clock::now();
            }
        }

        hctx.SynchronizeStreams();
    }

    void DoMuStep(DenseDenseMatMul& WHH_op, MultiplicativeUpdate& mu_op) {
        WHH_op.CallAsync();
        mu_op.CallAsync();
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // General purpose stuff.

    template <typename D, typename H>
    void CreateAndAllocDeviceMemory_(std::vector<D>& ds, std::vector<H>& hs) {
        // THROWSERT(hs.size() == n_devices);
        for(int i=0;i<hs.size();i++) {
            auto& h = hs[i];
            ds.emplace_back(h.n_rows, h.n_cols, h.nnz);
            ds[i].AllocMemory(dctx_main);
        }
    }

    template <typename D, typename H>
    void TransferToDevice_(std::vector<D>& ds, std::vector<H>& hs) {
        // THROWSERT(hs.size() == n_devices);
        for(int i=0;i<hs.size();i++) {
            dctx_main.CopyToDeviceAsync(ds[i], hs[i]);
        }
    }

    template <typename T>
    void FreeDeviceAllocs_Vector(std::vector<T>& v) {
        for(auto& x : v) { dctx_main.FreeDeviceAllocs(x); }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Another type of general purpose stuff.

    void SetMainDevice() {
        dctx_main.SetDevice();
    }

};



}
}
}
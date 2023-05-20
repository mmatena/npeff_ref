#ifndef Z099F5B1B_03A4_4638_81D3_D1CAA16678C8_H
#define Z099F5B1B_03A4_4638_81D3_D1CAA16678C8_H

/* Multiple GPU sparse data with dense factors and intermediates. 

CuSPARSE targets matrices with a number of (structural) zero elements
which represent > 95% of the total entries.

*/

#include <math.h>
#include <ctime>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <util/cuda_system.h>
#include <util/cuda_statuses.h>
#include <util/matrices.h>
#include <util/misc_util.h>
#include <util/matrix_util.h>
#include "./kernels.h"
#include "./multi_mu_dense_factors_device_contexts1.h"


#define _DEV__SYNC_FOR_TIMING() CUDA_CALL(cudaStreamSynchronize(stream))


///////////////////////////////////////////////////////////////////////////////


// TODO: Support column dimensions of greater than MAX_INT. Also probably make sure can support all sizes up to
// max_int when iterating over all the entries in a matrix.
 
template <typename IndT>
class MuNmf {
    // Will use all devices. Assumes that all devices are the same.
    // TODO: Check assertion that all devices are the same.

public:
    // MuNmf(ElCsrMatrix<IndT>* A, MuNmfParams p) {
    //     if (A->n_rows > A->n_cols) {
    //         // TODO: Handle this case.
    //         std::cout << "Error: Not handling matrices with more rows than columns yet.\n";
    //         throw;
    //     }
    //     this->max_iters = p.max_iters;
    //     this->eps = p.eps;
    //     this->seed = p.seed;
    //     // this->maxOutputMag = maxOutputMag;

    //     this->rank = p.rank;

    //     this->n_rows = A->n_rows;
    //     this->n_cols = A->n_cols;

    //     cudaGetDeviceCount(&nDevices);

    //     this->hostAPartitions = splitColumnWise(*A, nDevices);

    //     deviceContexts = (DeviceContext<IndT> *) malloc(nDevices * sizeof(DeviceContext<IndT>));
    //     for (int i=0; i<nDevices; i++) {
    //         MuNmfDeviceContextParams dcp;
    //         dcp.nmfParams = p;
    //         dcp.nDevices = nDevices;
    //         dcp.device = i;
    //         dcp.n_rows = n_rows;
    //         dcp.n_cols = n_cols;

    //         // TODO: THere is probably a better way than this.
    //         new(deviceContexts + i) DeviceContext<IndT>(hostAPartitions + i, dcp);
    //     }

    //     initializeNccl();
    // }

    MuNmf(ElCsrMatrix<IndT>* hostAPartitions, MuNmfParams p) {
        this->hostAPartitions = hostAPartitions;

        cudaGetDeviceCount(&nDevices);
        this->n_rows = hostAPartitions[0].n_rows;
        this->n_cols = 0;
        for (int i=0; i<nDevices; i++) {
            auto& a = hostAPartitions[i];
            if (a.n_rows != this->n_rows) {
                THROW;
            }
            this->n_cols += a.n_cols;
        }

        if (this->n_rows > this->n_cols) {
            // TODO: Handle this case.
            std::cout << "Error: Not handling matrices with more rows than columns yet.\n";
            throw;
        }
        this->max_iters = p.max_iters;
        this->eps = p.eps;
        this->seed = p.seed;
        // this->maxOutputMag = maxOutputMag;

        this->rank = p.rank;


        deviceContexts = (DeviceContext<IndT> *) malloc(nDevices * sizeof(DeviceContext<IndT>));
        for (int i=0; i<nDevices; i++) {
            MuNmfDeviceContextParams dcp;
            dcp.nmfParams = p;
            dcp.nDevices = nDevices;
            dcp.device = i;
            dcp.n_rows = n_rows;
            dcp.n_cols = n_cols;

            // TODO: THere is probably a better way than this.
            new(deviceContexts + i) DeviceContext<IndT>(hostAPartitions + i, dcp);
        }

        initializeNccl();
    }

    ~MuNmf() {
        // TODO:
    }

    void _initializeBeforeRun() {
        for (int i=0; i<nDevices; i++) {
            auto& dc = deviceContexts[i];
            if (i == 0) {
                // W is duplicated across devices, so we need to initialize once
                // and then broadcast across the devices.
                dc.initializeW(dc.devW);
            }

            dc.moveAToDeviceAsync();
            dc.initializeScalars();
            dc.initializeH(dc.devH);

            dc.computeASqNorm();

            // Allocates buffers and stuff for sparse-dense matrix multiplies.
            dc.preprocessSparseMatMuls();
        }

        broadcastInitializedW();
        synchronizeAllStreams();

    }

    void run() {
        _initializeBeforeRun();







        // std::cout << "JUST FOR INITIAL TESTING.\n";
        // QDBG(deviceContexts[0].sparseContextW->sparsifyF();)








        const int logLossFreq = 10;
        // const int logLossFreq = 1;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_iters; step++) {
            // TODO: Seems like we need to update W first sometimes to prevent some errors.
            // See if this is actual thing or a result of some bug.
            updateW();
            synchronizeAllStreams();

            updateH();
            synchronizeAllStreams();

            // NOTE: computing the loss after the W-update instead of after the
            // H update saves us an all-reduce. See if we can reverse the order of
            // the updates (probably fine due to symmetry of decomposition).
            if ((step + 1) % logLossFreq == 0) {
                auto t_end = std::chrono::high_resolution_clock::now();
                double elapsedMs = std::chrono::duration<double, std::milli>(t_end-t_start).count();

                // TODO: Compute loss only every k-steps. (Set as parameter).
                // float loss = computeLossAfterWUpdate();
                float loss = computeUncachedLoss();
                // TODO: Remove this.

                synchronizeAllStreams();
                // The ms/step ignores the time to compute the loss.
                std::cout << "step " << step + 1 << ": " << loss << " [" << elapsedMs / (double) logLossFreq << " ms/step]\n";

                t_start = std::chrono::high_resolution_clock::now();
            }

        }

    }

    MeMatrix loadWToHostSync() {
        MeMatrix W(n_rows, rank);

        // W is replicated across all devices, so only load from one of them.
        auto& dc = deviceContexts[0];

        CUDA_CALL(cudaSetDevice(dc.device));
        CUDA_CALL(
            cudaMemcpyAsync(W.data, dc.devW.devPtr, sizeof(float) * (long) W.n_rows * (long) W.n_cols, cudaMemcpyDeviceToHost, dc.stream)
        );
        CUDA_CALL(cudaStreamSynchronize(dc.stream));

        return W;
    }

    MeMatrix loadHToHostSync() {
        MeMatrix H(rank, n_cols);

        long offset = 0;
        for (int i = 0; i < nDevices; i++) {
            auto& dc = deviceContexts[i];
            CUDA_CALL(cudaSetDevice(dc.device));

            long matElements = (long) dc.devH.n_rows * (long) dc.devH.n_cols;

            CUDA_CALL(
                cudaMemcpyAsync(H.data + offset, dc.devH.devPtr, sizeof(float) * matElements, cudaMemcpyDeviceToHost, dc.stream)
            );
            offset += matElements;
        }
        synchronizeAllStreams();

        return H;
    }

    float computeUncachedLoss() {
        // Each device's contribution to the loss can be computed locally, with
        // only the need to return a couple floats from each device.

        for (int i = 0; i < nDevices; i++) {
            deviceContexts[i].computeUncachedLoss();
        }
        synchronizeAllStreams();

        float sqLoss = 0.0f;
        for (int i = 0; i < nDevices; i++) {
            auto& dc = deviceContexts[i];

            float posLoss = 0.0f;
            for(int j=0; j < dc.n_posLossChunks; j++) {
                posLoss += dc.posLoss[j];
            }

            float negLoss = 0.0f;
            for(int j=0; j < dc.n_negLossChunks; j++) {
                negLoss += dc.negLoss[j];
            }

            // std::cout << "Device " << i << ":\n";
            // std::cout << "    sq_A_norm: " << dc.local_sq_A_norm << "\n";
            // std::cout << "    posLoss: " << posLoss << "\n";
            // std::cout << "    negLoss: " << negLoss << "\n";

            sqLoss += dc.local_sq_A_norm + posLoss - 2 * negLoss;
        }

        return sqrt(sqLoss);
    }

private:
    int nDevices;
    DeviceContext<IndT>* deviceContexts;
    ncclComm_t* comms;

    int max_iters;
    float eps;

    long seed;

    int rank;

    int n_rows;
    int n_cols;

    ElCsrMatrix<IndT>* hostAPartitions;

    void updateW() {
        // Requires two allreduce-sum communications to compute AH^T and HH^T
        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].computeLocalAH();
            deviceContexts[i].computeLocalHH();
        }
        _updateW_allReduce();
        synchronizeAllStreams();
        
        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].updateWAfterAllReduces();
        }
    }

    void _updateW_allReduce() {
        NCCL_CALL(ncclGroupStart());
        for (int i = 0; i < nDevices; i++) {
            auto& ds = deviceContexts[i];
            NCCL_CALL(
                ncclAllReduce(
                    ds.devAFPtr,
                    ds.devAFPtr,
                    ds.n_rows * ds.rank,
                    ncclFloat,
                    ncclSum,
                    *(ds.comm),
                    ds.stream)
            );
            NCCL_CALL(
                ncclAllReduce(
                    ds.devFF.devPtr,
                    ds.devFF.devPtr,
                    ds.rank * ds.rank,
                    ncclFloat,
                    ncclSum,
                    *(ds.comm),
                    ds.stream)
            );
        }
        NCCL_CALL(ncclGroupEnd());
    }

    void updateH() {
        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].updateH();
        }
    }

    void initializeNccl() {
        comms = (ncclComm_t*) malloc(sizeof(ncclComm_t) * nDevices);
        
        NCCL_CALL(ncclCommInitAll(comms, nDevices, NULL));

        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].setComm(comms + i);
        }
    }

    void synchronizeAllStreams() {
        for (int i=0; i<nDevices; i++) {
            CUDA_CALL(cudaSetDevice(deviceContexts[i].device));
            CUDA_CALL(cudaStreamSynchronize(deviceContexts[i].stream));
        }
    }

    void broadcastInitializedW() {
        auto& dc = deviceContexts[0];
        NCCL_CALL(ncclGroupStart());
        for (int i=0; i<nDevices; i++) {
            NCCL_CALL(
                ncclBroadcast(
                    dc.devW.devPtr,
                    deviceContexts[i].devW.devPtr,
                    dc.devW.nEntries(),
                    ncclFloat,
                    dc.device,
                    *(deviceContexts[i].comm),
                    deviceContexts[i].stream
                )
            );
        }
        NCCL_CALL(ncclGroupEnd());
    }


};



#endif

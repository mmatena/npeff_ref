#ifndef Z4F939B37_1CC9_43EF_A74A_5784CAFBBDB0_H
#define Z4F939B37_1CC9_43EF_A74A_5784CAFBBDB0_H
/* Basic dense matrix multiplicative update NMF with support for multiple GPUs. */
#include <algorithm>
#include <iostream>
#include <math.h>

#include <curand.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nccl.h"

#include <util/cuda_statuses.h>
#include <util/matrices.h>
#include "multi_dense_mu_device_context.h"



// Get device infos
// Split matrices across devices.
// Run MU NMF.


class MuNmf {
    // Will use all devices. Assumes that all devices are the same.
    // TODO: Check assertion that all devices are the same.

public:
    MuNmf(MeMatrix* A, long rank, int max_iters, float eps, long seed) {
        if (A->n_rows > A->n_cols) {
            // TODO: Handle this case.
            throw;
        }
        this->max_iters = max_iters;
        this->eps = eps;
        this->seed = seed;

        this->rank = rank;

        this->n_rows = (long) A->n_rows;
        this->n_cols = (long) A->n_cols;

        this->hostA = A;

        // TODO: Probably put some stuf below in some method.

        cudaGetDeviceCount(&nDevices);

        if (n_cols % nDevices) {
            // TODO: Support cases where the number of columns is not a multiple
            // of the number of devices.
            throw;
        }

        deviceContexts = (DeviceContext *) malloc(nDevices * sizeof(DeviceContext));
        for (int i=0; i<nDevices; i++) {
            // TODO: THere is probably a better way than this.
            new(deviceContexts + i) DeviceContext(nDevices, i, seed, n_rows, n_cols, rank, eps);
        }


        initializeNccl();
    }

    ~MuNmf() {
        // TODO: Make sure everything is cleaned up.
        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].~DeviceContext();
        }
        for (int i=0; i<nDevices; i++) {
            ncclCommDestroy(comms[i]);
        }
        free(comms);
        free(deviceContexts);
    }

    void run() {
        // TODO: Multiple streams and possibly some reordering can make this faster.

        for (int i=0; i<nDevices; i++) {
            auto& dc = deviceContexts[i];
            // CUDA_CALL(cudaSetDevice(dc.device));
            if (i == 0) {
                // W is duplicated across devices, so we need to initialize once
                // and then broadcast across the devices.
                dc.initializeFactor(dc.devW);
            }

            dc.moveAToDeviceAsync(hostA);
            dc.initializeScalars();
            dc.initializeFactor(dc.devH);
        }

        broadcastInitializedW();
        synchronizeAllStreams();

        for (int step = 0; step < max_iters; step++) {
            updateW();
            synchronizeAllStreams();

            updateH();
            synchronizeAllStreams();

            float loss = computeLoss();

            // TODO: Remove this.
            std::cout << loss << '\n';

            synchronizeAllStreams();
        }
    }

private:

    void updateH() {
        for (int i=0; i<nDevices; i++) {
            deviceContexts[i].updateH();
        }
    }

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

    float computeLoss() {
        for (int i = 0; i < nDevices; i++) {
            deviceContexts[i].computeLoss();
        }
        synchronizeAllStreams();

        float sqLoss = 0.0f;
        for (int i = 0; i < nDevices; i++) {
            float localLoss = deviceContexts[i].localLoss;
            sqLoss += localLoss * localLoss;
        }
        return sqrt(sqLoss);
    }

private:
    int nDevices;
    DeviceContext* deviceContexts;
    ncclComm_t* comms;

    int max_iters;
    float eps;

    long seed;

    long rank;

    long n_rows;
    long n_cols;

    MeMatrix* hostA;

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

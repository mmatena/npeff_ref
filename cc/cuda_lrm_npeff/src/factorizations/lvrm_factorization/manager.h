#pragma once

#include <ctime>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <iomanip>

#include "nccl.h"

#include <util/macros.h>
#include <gpu/macros.h>
#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include "./config.h"
#include "./device_worker.h"


namespace npeff {
namespace factorizations {
namespace lvrm_factorization {


template<typename IndT>
class MultiGpuManager {
    using DeviceWorkerPtr = std::unique_ptr<DeviceWorker<IndT>>;
    using CsrMatrixPtr = std::unique_ptr<npeff::CsrMatrix<IndT>>;
    using DenseMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;

    FactorizationConfig config;
    int64_t n_partitions;

    // example_row_offsets.shape = [1, n_examples + 1].
    std::unique_ptr<npeff::DenseMatrix<int64_t>> example_row_offsets;

    // The array of comms used for nccl.
    std::unique_ptr<ncclComm_t> comms;

    // A vector of workers. Each is assigned to a unique device.
    std::vector<DeviceWorkerPtr> workers;

    // Information about the size of the matrix and its partitions.
    const int64_t n_examples;
    const int64_t n_cols_total;
    const std::vector<int64_t> n_cols_per_partition;

public:
    
    // Collects losses whenever they get computed.
    std::vector<float> losses_G_only;
    std::vector<float> losses_joint;

    MultiGpuManager(
        std::vector<CsrMatrixPtr>& column_partitions,
        FactorizationConfig config,
        std::unique_ptr<npeff::DenseMatrix<int64_t>> example_row_offsets
    ) :
        config(config),
        example_row_offsets(std::move(example_row_offsets)),
        n_partitions(column_partitions.size()),
        n_examples(config.n_examples),
        n_cols_total(compute_n_cols_total(column_partitions)),
        n_cols_per_partition(compute_n_cols_per_partition(column_partitions))
    {
        initialize_nccl();
        create_workers(column_partitions);
    }

    ~MultiGpuManager() {
        // Have this check in place so that we do not try to
        // pass junk addresses to ncclCommDestroy in case we
        // did not allocate the comms array at the time of
        // desctruction.
        if(comms) {
            for (int64_t i=0; i<n_partitions; i++) {
                ncclCommDestroy(comms.get()[i]);
            }
        }
    }

    
    DenseMatrixPtr read_W_from_gpu(int64_t src_device = 0) {
        DeviceWorkerPtr& src_worker = workers[src_device];
        DenseMatrixPtr W = std::unique_ptr<DenseMatrix<float>>(
            new DenseMatrix<float>(n_examples, config.rank));
        src_worker->read_W_from_gpu_async(W->data.get());
        src_worker->synchronize_stream();
        return W;
    }

    DenseMatrixPtr read_G_from_gpu() {
        DenseMatrixPtr G = std::unique_ptr<DenseMatrix<float>>(
            new DenseMatrix<float>(config.rank, n_cols_total));

        float* data = G->data.get();
        for(int64_t i=0; i<n_partitions; i++) {
            workers[i]->read_G_from_gpu_async(data);
            data += config.rank * n_cols_per_partition[i];
        }

        synchronize_all_streams();
        return G;
    }

    /////////////////////////////////////////////////////////////////
    // Highest level function for "main loop" of optimization.

    void run() {
        get_workers_ready();
        run_G_only();
        run_joint();
    }

protected:

    // For computing step times.
    std::chrono::high_resolution_clock::time_point t_step_start;

    void run_G_only() {
        t_step_start = std::chrono::high_resolution_clock::now();
        for (int64_t step = 0; step < config.n_iters_G_only; step++) {
            G_update_step(true, config.learning_rate_G_G_only);
            if(should_log_loss_at_step(step)) {
                log_loss("G only", step, losses_G_only);
            }
        }
    }

    void run_joint() {
        t_step_start = std::chrono::high_resolution_clock::now();
        for (int64_t step = 0; step < config.n_iters_joint; step++) {
            W_update_step(true);
            G_update_step(false, config.learning_rate_G_joint);
            if(should_log_loss_at_step(step)) {
                log_loss("joint", step, losses_joint);
            }
        }
    }

    bool should_log_loss_at_step(int64_t step) {
        return (step + 1) % config.log_loss_frequency == 0;
    }

    void log_loss(const std::string& prefix, int64_t step, std::vector<float>& losses) {
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end-t_step_start).count();

        // NOTE: For `run_joint`, I think we could save the all-reduces
        // computing the loss after the W update step. For simplicity
        // and robustness, I'm always doing the all-reduces for now.
        double loss = compute_loss(true);
        losses.push_back(loss);

        std::cout << prefix << " step " << step + 1 << ": " << loss << " [" << elapsed_ms / (double) config.log_loss_frequency << " ms/step]\n";

        t_step_start = std::chrono::high_resolution_clock::now();
    }

    /////////////////////////////////////////////////////////////////
    // Set up and sub-steps of the optimization.

    void get_workers_ready() {
        call_on_workers_then_join(&DeviceWorker<IndT>::set_up_work);

        // Replicate the W from device 0 on all of the devices. This
        // ensures consistency of the shared parameters across devices.
        broadcast_W_async();
        synchronize_all_streams();
    }

    void W_update_step(bool recompute_AG_GG) {
        if(recompute_AG_GG) {
            call_on_workers_then_join(&DeviceWorker<IndT>::compute_local_AG_GG_async);
            all_reduce_AG_GG_async();
        }
        call_on_workers_then_join(&DeviceWorker<IndT>::update_local_W_after_all_reduces_async);
        synchronize_all_streams();
    }

    void G_update_step(bool recompute_AG_GG, float learning_rate_G) {
        if(recompute_AG_GG) {
            call_on_workers_then_join(&DeviceWorker<IndT>::compute_local_AG_GG_async);
            all_reduce_AG_GG_async();
        }
        call_on_workers_then_join(&DeviceWorker<IndT>::update_local_G_after_all_reduces_async, learning_rate_G);
        synchronize_all_streams();
    }

    float compute_loss(bool recompute_AG_GG) {
        if(recompute_AG_GG) {
            call_on_workers_then_join(&DeviceWorker<IndT>::compute_local_AG_GG_async);
            all_reduce_AG_GG_async();
        }
        // We only need to compute the loss using a single GPU.
        auto& loss_worker = workers[0];
        loss_worker->compute_loss_after_all_reduces_async();
        float loss_term = loss_worker->read_loss_term_from_device();
        synchronize_all_streams();
        return loss_term + config.tr_xx;
    }

    /////////////////////////////////////////////////////////////////
    // Initializations.

    int64_t compute_n_cols_total(std::vector<CsrMatrixPtr>& column_partitions) {
        int64_t n_cols = 0;
        for(auto& mat : column_partitions) { n_cols += mat->n_cols; }
        return n_cols;
    }

    std::vector<int64_t> compute_n_cols_per_partition(std::vector<CsrMatrixPtr>& column_partitions) {
        std::vector<int64_t> ret;
        for(auto& mat : column_partitions) {
            ret.push_back(mat->n_cols);
        }
        return ret;
    }

    void initialize_nccl() {
        comms = std::unique_ptr<ncclComm_t>(new ncclComm_t[n_partitions]);
        NCCL_CALL(ncclCommInitAll(comms.get(), n_partitions, NULL));
    }

    void create_workers(std::vector<CsrMatrixPtr>& column_partitions) {
        for (int64_t i=0; i<n_partitions; i++) {
            workers.push_back(
                DeviceWorkerPtr(new DeviceWorker<IndT>(
                    std::move(column_partitions[i]),
                    config,
                    example_row_offsets.get(),
                    i,
                    n_partitions,
                    comms.get()[i]
                )));
        }
    }

    /////////////////////////////////////////////////////////////////
    // Utilities for dealing the multiple workers.

    void broadcast_W_async(int64_t src_device = 0) {
        DeviceWorkerPtr& src_worker = workers[src_device];
        NCCL_CALL(ncclGroupStart());
        for(auto& worker : workers) {
            worker->nccl_broadcast_of_W(*src_worker);
        }
        NCCL_CALL(ncclGroupEnd());

    }

    void all_reduce_AG_GG_async() {
        NCCL_CALL(ncclGroupStart());
        for(auto& worker : workers) {
            worker->nccl_all_reduce_AG_GG();
        }
        NCCL_CALL(ncclGroupEnd());
    }

    void synchronize_all_streams() {
        for(auto& worker : workers) {
            worker->synchronize_stream();
        }
    }

    void join_threads(std::vector<std::thread>& threads) {
        for (auto& thread : threads) { thread.join(); }
    }

    template<typename... Args>
    void call_on_workers_then_join(void (DeviceWorker<IndT>::*method)(Args...), Args... args) {
        std::vector<std::thread> threads;
        for(auto& worker : workers) {
            threads.emplace_back(method, worker.get(), args...);
        }
        join_threads(threads);
    }

};


}  // lvrm_factorization
}  // factorizations
}  // npeff

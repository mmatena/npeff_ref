#pragma once
/**
 * Ops for getting e.g. a contiguous subset of columns of a sparse matrix.
 */

#include <algorithm>
#include <vector>
#include <tuple>
#include <utility>
#include <memory>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>
#include <misc/macros.h>

#include <cuda/cuda_context.h>
#include <cuda/descr.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>

#include <cuda/ops/transpose.h>
#include <cuda/ops/matrix_conversion.h>




namespace Cuda {
namespace Ops {


// Forward declarations:
template <typename IndT_In, typename IndT_Out>
class SplitByCols;


struct SplitInfo {
    long start_index;
    long size;

    // Note: Won't always be set.
    long nnz = -1;

    SplitInfo(long start_index, long size) :
        start_index(start_index), size(size)
    {}

    SplitInfo(long start_index, long size, long nnz) :
        start_index(start_index), size(size), nnz(nnz)
    {}
};


namespace internal {

__host__ __device__ long GetSizeOfSplit(long total_size, long n_splits, long split_index) {
    // The last split will be the one with potentially a few extra elements.
    long baseline_size = total_size / n_splits;
    if(split_index == n_splits - 1) {
        return baseline_size + (total_size % n_splits);
    } else {
        return baseline_size;
    }
}

std::vector<SplitInfo> MakeSplitInfos_Rows(long total_size, long n_splits) {
    std::vector<SplitInfo> ret;

    long start_index = 0;
    for(long i=0;i<n_splits;i++) {
        long size = GetSizeOfSplit(total_size, n_splits, i);
        ret.emplace_back(start_index, size);
        start_index += size;
    }

    return ret;
}



// NOTE: Trying some dynamic parallelism stuff.


template <typename IndT>
__global__
void SplitByRows_InPlace_UpdateRowsKernel(long n_row_offsets_in_split, const IndT* split_row_offsets_start, IndT* new_split_row_offsets_start) {
    IndT og_start_offset = split_row_offsets_start[0];
    INDEX_STRIDE_1D(n_row_offsets_in_split, i) {
        new_split_row_offsets_start[i] = split_row_offsets_start[i] - og_start_offset;
    }
}


template <typename IndT, long block_size>
__global__
void SplitByRows_InPlace_LaunchingKernel(long n_splits, long n_total_rows, const IndT* row_offsets, IndT* new_row_offsets) {
    long baseline_size = n_total_rows / n_splits;

    INDEX_STRIDE_1D(n_splits, split_index) {

        long start_row_index = split_index * baseline_size;

        long n_rows_in_split = GetSizeOfSplit(n_total_rows, n_splits, split_index);
        long n_row_offsets_in_split = 1 + n_rows_in_split;

        long n_blocks = (n_row_offsets_in_split + block_size - 1) / block_size;

        SplitByRows_InPlace_UpdateRowsKernel<IndT><<<n_blocks, block_size>>>(
            n_row_offsets_in_split,
            row_offsets + start_row_index,
            new_row_offsets + start_row_index + split_index);

    }
}


template <typename IndT>
struct SplitByRows_DeviceHelper {
    long n_splits;
    long n_total_rows;

    // These all point to memory on the GPU.
    SplitInfo* split_infos;
    IndT* row_offsets;
    IndT* out_row_offsets;

    void CallAsync(DeviceCudaContext& ctx, long block_size); 
};


template <typename IndT>
__global__ 
void UpdateRows_Kernel(SplitByRows_DeviceHelper<IndT> h, const SplitInfo info, const long split_index) {
    IndT og_start_offset = h.row_offsets[info.start_index];
    long n_row_offsets = info.size + 1;
    INDEX_STRIDE_1D(n_row_offsets, i) {
        h.out_row_offsets[info.start_index + split_index + i] = h.row_offsets[info.start_index + i] - og_start_offset;
    }
}


template <typename IndT>
__global__
void Launch_Kernel(SplitByRows_DeviceHelper<IndT> h, const long block_size) {
    INDEX_STRIDE_1D(h.n_splits, split_index) {
        const SplitInfo info = h.split_infos[split_index];
        long n_blocks = (info.size + block_size) / block_size;
        UpdateRows_Kernel<IndT><<<n_blocks, block_size>>>(h, info, split_index);
    }
}

template <typename IndT>
void SplitByRows_DeviceHelper<IndT>::CallAsync(DeviceCudaContext& ctx, long block_size) {
    long n_blocks = (n_splits + block_size - 1) / block_size;
    Launch_Kernel<IndT><<<n_blocks, block_size, 0, ctx.stream>>>(*this, block_size);
}


} // internal



enum SplitUniformity {USER, ROWS, NNZ};


template <typename IndT>
class SplitByRows_InPlace {
    using CsrMatrix = Device::CsrMatrix<IndT>;

    template<typename IndT_In, typename IndT_Out>
    friend class SplitByCols;

protected:
    DeviceCudaContext& ctx;
    CsrMatrix& mat;

    long n_splits;

    SplitUniformity uniformity;

    std::vector<SplitInfo> split_infos;

    // Length = n_splits.
    IndT* og_split_end_offsets = nullptr;



    // TODO: I can't actually do this op in place, so figure out how to handle this properly.
    IndT* new_row_offsets = nullptr;



    // Constructor to be called from other slicing classes (must be friends).
    SplitByRows_InPlace(
        DeviceCudaContext& ctx,
        CsrMatrix& mat,
        std::vector<SplitInfo>& split_infos,
        long n_splits,
        SplitUniformity uniformity
    ) :
        ctx(ctx), mat(mat), n_splits(n_splits), uniformity(uniformity),
        split_infos(split_infos),
        og_split_end_offsets(new IndT[n_splits])
    {
        if(uniformity == USER) {
            this->n_splits = split_infos.size();
            THROWSERT(split_infos.back().start_index + split_infos.back().size == mat.n_rows);
        } else {
            THROWSERT(split_infos.size() == 0);
            this->split_infos = ComputeSplitInfos();
        }
    }


public:
    SplitByRows_InPlace(
        DeviceCudaContext& ctx,
        CsrMatrix& mat,
        long n_splits,
        SplitUniformity uniformity = ROWS
    ) :
        ctx(ctx), mat(mat), n_splits(n_splits), uniformity(uniformity),
        split_infos(ComputeSplitInfos()),
        og_split_end_offsets(new IndT[n_splits])
    {}

    SplitByRows_InPlace(
        DeviceCudaContext& ctx,
        CsrMatrix& mat,
        std::vector<SplitInfo>& split_infos
    ) :
        ctx(ctx), mat(mat), n_splits(split_infos.size()), uniformity(USER),
        split_infos(split_infos),
        og_split_end_offsets(new IndT[n_splits])
    {
        THROWSERT(split_infos.back().start_index + split_infos.back().size == mat.n_rows);
    }

    ~SplitByRows_InPlace() {
        delete[] og_split_end_offsets;
    }

    // NOTE: Not really async.
    void CallAsync() {
        ctx.SetDevice();


        new_row_offsets = ctx.dmalloc<IndT>(mat.n_rows + n_splits);


        // TODO: Read the specific offset ranges of the splits into host memory
        for(long i=0;i<n_splits;i++) {
            auto& info = split_infos[i];
            long end_index = info.start_index + info.size;

            ctx.CopyToHostAsync(og_split_end_offsets + i, mat.row_offsets + end_index, 1);
        }

        SplitInfo* device_infos = ctx.dmalloc<SplitInfo>(n_splits);
        ctx.CopyToDeviceAsync<SplitInfo>(device_infos, split_infos.data(), n_splits);

        internal::SplitByRows_DeviceHelper<IndT> h;
        h.n_splits = n_splits;
        h.n_total_rows = mat.n_rows;
        h.split_infos = device_infos;
        h.row_offsets = mat.row_offsets;
        h.out_row_offsets = new_row_offsets;

        const long block_size = 512;
        h.CallAsync(ctx, block_size);
       
        ctx.SynchronizeStream();
        ctx.dfree(device_infos);
    }

    // NOTE: Need to call stream synch before calling this.
    std::vector<CsrMatrix> GetSplits() {
        ctx.SetDevice();

        std::vector<CsrMatrix> ret;

        for(long i=0;i<n_splits;i++) {
            auto& info = split_infos[i];

            IndT og_start_offset = i == 0 ? 0 : og_split_end_offsets[i - 1];
            IndT og_end_offset = og_split_end_offsets[i];
            long nnz = og_end_offset - og_start_offset;

            ret.emplace_back(info.size, mat.n_cols, nnz);
            auto& split = ret[i];
            split.values = mat.values + og_start_offset;
            split.col_indices = mat.col_indices + og_start_offset;
            split.row_offsets = new_row_offsets + (info.start_index + i);
            split.CreateDescr();
        }

        return ret;
    }

protected:
    std::vector<SplitInfo> ComputeSplitInfos() {
        if(uniformity == ROWS) {
            return internal::MakeSplitInfos_Rows(mat.n_rows, n_splits);
        } else if(uniformity == NNZ) {
            return ComputeSplitInfos_Nnz();
        } else {
            THROW;
        }
    }

    std::vector<SplitInfo> ComputeSplitInfos_Nnz() {
        long n_rows = mat.n_rows;

        IndT* row_offsets = new IndT[n_rows + 1];
        ctx.CopyToHostAsync(row_offsets, mat.row_offsets, n_rows + 1);
        ctx.SynchronizeStream();

        std::vector<SplitInfo> infos;
        long expected_split_nnz = mat.nnz / n_splits;

        long current_split_nnz = 0;
        for(long i=0;i<n_rows;i++) {
            long row_nnz = row_offsets[i + 1] - row_offsets[i];
            current_split_nnz += row_nnz;

            if (current_split_nnz >= expected_split_nnz || i == n_rows - 1) {
                if (infos.size() == 0) {
                    infos.emplace_back(0, i + 1, current_split_nnz);
                } else {
                    auto& back = infos.back();
                    long start_index = back.start_index + back.size;
                    long split_size = i + 1 - start_index;
                    infos.emplace_back(start_index, split_size, current_split_nnz);
                }
                current_split_nnz = 0;
            }
        }

        // TODO: Maybe now join splits with zero nnz with neighbors? Choose
        // the neighbor with the least rows.

        // Now split the largest k groups in half until we have
        // the correct number of splits. We should never have
        // more than the requested number of splits since each of the
        // original splits (except maybe the last one) has size greater
        // than the required average
        // size of split.

        long need_to_add = n_splits - infos.size();
        THROWSERT(need_to_add >= 0);

        if (need_to_add > 0) {
            std::vector<SplitInfo> infos_copy(infos);

            auto cmp = [](SplitInfo& a, SplitInfo& b) {
                // Make splits consisting of a single row behave as if they have the smallest
                // nnz as we cannot subdivide them further.
                long a_nnz = a.size == 1 ? -1 : a.nnz;
                long b_nnz = b.size == 1 ? -1 : b.nnz;
                return a_nnz > b_nnz;
            };
            std::nth_element(
                infos_copy.begin(),
                infos_copy.end(),
                infos_copy.begin() + need_to_add,
                cmp);


            std::vector<SplitInfo> top_infos_by_nnz(infos_copy.begin(), infos_copy.begin() + need_to_add);

            std::vector<SplitInfo> new_infos;
            for(auto& info : infos) {

                bool needs_split = false;
                for(auto& top_info : top_infos_by_nnz) {
                    if (top_info.start_index == info.start_index) {
                        needs_split = true;
                        break;
                    }
                }

                if(needs_split) {
                    auto bifurc = BifurcateSplit_Nnz(info, row_offsets);
                    new_infos.push_back(bifurc.first);
                    new_infos.push_back(bifurc.second);
                } else {
                    new_infos.push_back(info);
                }

            }

            std::swap(infos, new_infos);
        }

        delete[] row_offsets;

        THROWSERT(infos.size() == n_splits);

        return infos;
    }

    std::pair<SplitInfo, SplitInfo> BifurcateSplit_Nnz(const SplitInfo& info, const IndT* row_offsets) {
        std::vector<long> cumsum = {0};
        for(int i=info.start_index; i < info.start_index + info.size; i++) {
            cumsum.push_back(cumsum.back() + row_offsets[i+1] - row_offsets[i]);
        }

        // Choose index of cumsum closest to half of info.nnz.
        long desired_nnz = info.nnz / 2;

        long closest_idx = -1;
        long smallest_abs_value = LONG_MAX;

        for(int i=0;i<cumsum.size();i++) {
            long abs_value = std::abs(cumsum[i] - desired_nnz);
            if (abs_value < smallest_abs_value) {
                closest_idx = i;
                smallest_abs_value = abs_value;
            }
        }

        SplitInfo first(info.start_index, closest_idx, cumsum[closest_idx]);
        SplitInfo second(info.start_index + closest_idx, info.size - closest_idx, info.nnz - cumsum[closest_idx]);

        return std::make_pair(first, second);
    }

};


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////



template <typename IndT_In, typename IndT_Out = IndT_In>
class SplitByCols {
    using CsrMatrix_In = Device::CsrMatrix<IndT_In>;
    using CsrMatrix_Out = Device::CsrMatrix<IndT_Out>;

    using Custom_CsrTranspose_In = Custom_CsrTranspose<IndT_In>;
    using Custom_CsrTranspose_Out = Custom_CsrTranspose<IndT_Out>;

protected:
    DeviceCudaContext& ctx;
    CsrMatrix_In& mat;

    long n_splits;

    SplitUniformity uniformity;

    // After calling the Call method, this will always be filled in. Before that,
    // it will only be filled if the uniformity is USER.
    std::vector<SplitInfo> split_infos;

public:
    SplitByCols(
        DeviceCudaContext& ctx,
        CsrMatrix_In& mat,
        long n_splits,
        SplitUniformity uniformity = ROWS
    ) :
        ctx(ctx), mat(mat), n_splits(n_splits), uniformity(uniformity)
    {}

    SplitByCols(
        DeviceCudaContext& ctx,
        CsrMatrix_In& mat,
        std::vector<SplitInfo>& split_infos
    ) :
        ctx(ctx), mat(mat), n_splits(split_infos.size()), uniformity(USER),
        split_infos(split_infos)
    {
        THROWSERT(split_infos.back().start_index + split_infos.back().size == mat.n_cols);
    }

    std::vector<SplitInfo> GetSplitInfos() {
        return split_infos;
    }


    // TODO: Allow index type conversions to operate on the transposed_splits
    // so that we can use the int32 version of the transpose back.

    CsrMatrix_Out ConvertMatrixType(CsrMatrix_In& in) {
        // Only compiles if IndT_Out = int32_t.
        bool free_memory = false;
        ReIndexWithInt32<IndT_In> reindex_op(ctx, in, free_memory);
        reindex_op.Call();
        return std::move(reindex_op.out);
    }


    // This works by transposing, splitting by rows, and then transposing each split.
    std::vector<CsrMatrix_Out> Call() {
        ctx.SetDevice();

        // Allocate some temporary memory.
        CsrMatrix_In transposed_mat(mat.n_cols, mat.n_rows, mat.nnz);
        transposed_mat.AllocMemory(ctx);
        // std::cout << "Allocated memory for first transpose of col slicing.\n";

        // Perform the transpose of the original matrix.
        Custom_CsrTranspose_In transpose_op(ctx, mat, transposed_mat);
        transpose_op.Call_SingleUse();
        // std::cout << "First transpose of col slicing.\n";

        // Split the transposed matrix by its rows, which correspond to
        // columns in the original matrix.
        // SplitByRows_InPlace<IndT> split_by_rows_op(ctx, transposed_mat, n_splits, uniformity);
        SplitByRows_InPlace<IndT_In> split_by_rows_op(
            ctx, transposed_mat, split_infos, n_splits, uniformity);
        split_by_rows_op.CallAsync();
        ctx.SynchronizeStream();
        this->split_infos = split_by_rows_op.split_infos;
        std::vector<CsrMatrix_In> transposed_splits = split_by_rows_op.GetSplits();

        // std::cout << "Split by rows completed.\n";

        // Transpose the transposed splits to get our final splits.
        std::vector<CsrMatrix_Out> splits;
        for(long i=0;i<n_splits;i++) {
            auto& t_split = transposed_splits[i];
            splits.emplace_back(t_split.n_cols, t_split.n_rows, t_split.nnz);

            auto& split = splits[i];
            split.AllocMemory(ctx);

            CsrMatrix_Out t_split2 = ConvertMatrixType(t_split);
            Custom_CsrTranspose_Out t_split_op(ctx, t_split2, split);
            // Custom_CsrTranspose_Out t_split_op(ctx, t_split, split);
            t_split_op.Call_SingleUse();

            if (!std::is_same<IndT_In, IndT_Out>::value) {
                ctx.dfree(t_split2.row_offsets);
                ctx.dfree(t_split2.col_indices);
            }
        }

        // Free the temporary memory.
        ctx.FreeDeviceAllocs(transposed_mat);

        return splits;
    }

};


// template<typename IndT>
// SplitByCols<IndT, IndT>::CsrMatrix_Out
// SplitByCols<IndT, IndT>::ConvertMatrixType(SplitByCols<IndT, IndT>::CsrMatrix_In& in) {
//     return in;
// }

// template<typename IndT_In>
// Device::CsrMatrix<int32_t>
// SplitByCols<IndT_In, int32_t>::ConvertMatrixType(Device::CsrMatrix<IndT_In>& in) {
//     bool free_memory = false;
//     ReIndexWithInt32<IndT_In> reindex_op(ctx, in, free_memory);
//     reindex_op.Call();
//     return std::move(reindex_op.out);
// }


} // Ops
} // Cuda


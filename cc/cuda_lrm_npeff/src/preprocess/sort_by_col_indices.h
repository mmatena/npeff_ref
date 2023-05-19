#pragma once
// Sorts the col_indices and values in place of a CsrMatrix
// such that col_indices are in ascending order. This might
// improve SpDn matmul performance, and it makes it easier
// to compute the tr_XX term on the CPU.

#include <algorithm>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include <containers/sparse_matrix.h>

namespace npeff {
namespace preprocessing {


namespace internal_ {

template <typename IndT>
class ColIndexSorter {
    CsrMatrix<IndT>& matrix;
    const int64_t chunk_index;
    const int64_t n_chunks;

    const int64_t start_row_index;
    const int64_t end_row_index;

    std::vector<float> values_buffer;
    std::vector<IndT> col_indices_buffer;
    std::vector<IndT> argsort_buffer;

public:
    ColIndexSorter(CsrMatrix<IndT>& matrix, int64_t chunk_index, int64_t n_chunks) :
        matrix(matrix), chunk_index(chunk_index), n_chunks(n_chunks),
        start_row_index(get_start_row_index()), end_row_index(get_end_row_index())
    {}

    void operator()() {
        int64_t max_row_size = compute_max_row_size();

        values_buffer = std::vector<float>(max_row_size);
        col_indices_buffer = std::vector<IndT>(max_row_size);
        argsort_buffer = std::vector<IndT>(max_row_size);

        for(int64_t i=start_row_index; i<end_row_index; i++) {
            sort_for_row(i);
        }
    }

protected:

    void sort_for_row(int64_t row_index) {
        float* values = matrix.values.get();
        IndT* row_offsets = matrix.row_offsets.get();
        IndT* col_indices = matrix.col_indices.get();

        int64_t start_index = row_offsets[row_index];
        int64_t end_index = row_offsets[row_index + 1];
        int64_t row_size = end_index - start_index;

        std::iota(argsort_buffer.begin(), argsort_buffer.begin() + row_size, 0);
        std::sort(argsort_buffer.begin(), argsort_buffer.begin() + row_size,
              [&col_indices, start_index](int left, int right) -> bool {
                  return col_indices[start_index + left] < col_indices[start_index + right];
              });

        std::copy(values + start_index, values + end_index, values_buffer.begin());
        std::copy(col_indices + start_index, col_indices + end_index, col_indices_buffer.begin());

        for(int64_t i=0; i<row_size; i++) {
            values[start_index + i] = values_buffer[argsort_buffer[i]];
            col_indices[start_index + i] = col_indices_buffer[argsort_buffer[i]];
        }
    }


    int64_t get_start_row_index() {
        return chunk_index * (matrix.n_rows / n_chunks);
    }
    int64_t get_end_row_index() {
        if (chunk_index == n_chunks - 1) {
            return matrix.n_rows;
        } else {
            return (chunk_index + 1) * (matrix.n_rows / n_chunks);
        }
    }

    int64_t compute_max_row_size() {
        IndT* row_offsets = matrix.row_offsets.get();
        int64_t max_size = 0;
        for(int64_t i=start_row_index; i<end_row_index; i++) {
            int64_t row_size = row_offsets[i + 1] - row_offsets[i];
            if (row_size > max_size) {
                max_size = row_size;
            }
        }
        return max_size;
    }
};

}  // internal_


template <typename IndT>
void sort_by_col_indices(CsrMatrix<IndT>* matrix, int64_t n_threads) {
    std::vector<internal_::ColIndexSorter<IndT>> col_sorters;
    for(int64_t i=0; i<n_threads; i++) {
        col_sorters.emplace_back(*matrix, i, n_threads);
    }

    std::vector<std::thread> threads;
    for(auto& sorter : col_sorters) {
        threads.emplace_back(sorter);
    }

    for(auto& thread : threads) {
        thread.join();
    }
}



}  // preprocessing
}  // npeff

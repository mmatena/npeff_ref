#pragma once
// Computes tr_XX on the CPU using multiple threads.

#include <thread>
#include <vector>

#include <containers/sparse_matrix.h>

namespace npeff {
namespace factorization {


namespace internal_ {

template <typename IndT>
class TrXxComputer {
    CsrMatrix<IndT>& matrix;
    const int64_t n_classes;

    const int64_t chunk_index;
    const int64_t n_chunks;

    const int64_t n_examples;

public:
    double tr_xx_contribution = -1.0;

    TrXxComputer(CsrMatrix<IndT>& matrix, int64_t n_classes, int64_t chunk_index, int64_t n_chunks) :
        matrix(matrix), n_classes(n_classes), chunk_index(chunk_index), n_chunks(n_chunks),
        n_examples(matrix.n_rows / n_classes)
    {
        THROW_IF_FALSE((matrix.n_rows % n_classes) == 0);
    }

    void operator()() {
        int64_t start_example_index = get_start_example_index();
        int64_t end_example_index = get_end_example_index();

        this->tr_xx_contribution = 0.0;
        for(int64_t i=start_example_index; i<end_example_index; i++) {
            this->tr_xx_contribution += compute_contribution_from_example(i);
        }
    }

protected:

    double compute_contribution_from_example(int64_t example_index) {
        double ret = 0.0;
        for(int64_t i=0; i<n_classes; i++) {
            double diag_term = compute_diagonal_term(example_index, i);
            ret += diag_term * diag_term;
            for(int64_t j=0; j<n_classes; j++) {
                double off_diag_term = compute_off_diagonal_term(example_index, i, j);                
                ret += 2.0 * off_diag_term * off_diag_term;
            }
        }
        return ret;
    }

    double compute_diagonal_term(int64_t example_index, int64_t class_index) {
        float* values = matrix.values.get();

        int64_t row_index = get_row_index(example_index, class_index);

        int64_t row_start_index = matrix.row_offsets.get()[row_index];
        int64_t row_end_index = matrix.row_offsets.get()[row_index + 1];

        double term = 0.0;
        for(int64_t i=row_start_index; i<row_end_index; i++) {
            term += values[i] * values[i];
        }

        return term;
    }

    double compute_off_diagonal_term(int64_t example_index, int64_t ci1, int64_t ci2) {
        float* values = matrix.values.get();
        IndT* row_offsets = matrix.row_offsets.get();
        IndT* col_indices = matrix.col_indices.get();

        int64_t ri1 = get_row_index(example_index, ci1);
        int64_t r1_end_offset = row_offsets[ri1 + 1];

        int64_t ri2 = get_row_index(example_index, ci2);
        int64_t r2_end_offset = row_offsets[ri2 + 1];

        int64_t col1_offset = row_offsets[ri1];
        int64_t col2_offset = row_offsets[ri2];
        double term = 0.0;
        while(col1_offset < r1_end_offset && col2_offset < r2_end_offset) {
            IndT col1 = col_indices[col1_offset];
            IndT col2 = col_indices[col2_offset];
            if(col1 == col2) {
                term += values[col1_offset] * values[col2_offset];
                col1_offset++;
                col2_offset++;
            } else if(col1 < col2) {
                col1_offset++;
            } else {
                col2_offset++;
            }
        }
        return term;
    }

    int64_t get_start_example_index() {
        return chunk_index * (n_examples / n_chunks);
    }
    int64_t get_end_example_index() {
        if (chunk_index == n_chunks - 1) {
            return n_examples;
        } else {
            return (chunk_index + 1) * (n_examples / n_chunks);
        }
    }

    int64_t get_row_index(int64_t example_index, int64_t class_index) {
        return example_index * n_classes + class_index;
    }

};

}  // internal_


// NOTE: Each row of the matrix must already have its col_indices sorted in ascending order.
template <typename IndT>
double compute_tr_xx(CsrMatrix<IndT>& matrix, int64_t n_classes, int64_t n_threads) {
    std::vector<internal_::TrXxComputer<IndT>> workers;
    for(int64_t i=0; i<n_threads; i++) {
        workers.emplace_back(matrix, n_classes, i, n_threads);
    }

    std::vector<std::thread> threads;
    for(auto& worker : workers) {
        threads.emplace_back(std::ref(worker));
    }

    for(auto& thread : threads) {
        thread.join();
    }

    double tr_xx = 0.0;
    for(auto& worker : workers) {
        tr_xx += worker.tr_xx_contribution;
    }

    return tr_xx;
}



}  // factorization
}  // npeff


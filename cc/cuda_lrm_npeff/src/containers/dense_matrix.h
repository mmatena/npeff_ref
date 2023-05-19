#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include <util/array_util.h>

namespace npeff {

// The layout of the data is assumed to be column major.
template<typename T>
struct DenseMatrix {
    int64_t n_rows;
    int64_t n_cols;

    std::unique_ptr<T> data;

    // Derived attributes.    
    int64_t n_entries;
    size_t size_bytes;

    DenseMatrix(int64_t n_rows, int64_t n_cols) :
        DenseMatrix(n_rows, n_cols, std::unique_ptr<T>(new T[n_rows * n_cols]))
    {}

    DenseMatrix(int64_t n_rows, int64_t n_cols, std::unique_ptr<T> data) :
        n_rows(n_rows),
        n_cols(n_cols),
        data(std::move(data)),
        n_entries(n_rows * n_cols),
        size_bytes(n_entries * sizeof(T))
    {}

    T* get_col_ptr(int64_t col_index) {
        return data.get() + (col_index * n_rows);
    }

    T get_entry(int64_t row_index, int64_t col_index) {
        return data.get()[col_index * n_rows + row_index];
    }

    bool operator==(const DenseMatrix<T> &o) const {
        return n_rows == o.n_rows && n_cols == o.n_cols && util::arrays_are_equal<T>(data.get(), o.data.get(), n_entries);
    }

    void transpose_in_place() {
        convert_to_row_major_in_place();
        std::swap(n_rows, n_cols);
    }

    void convert_to_row_major_in_place() {
        // TODO: Maybe change this into an actually in-place algorithm.
        auto new_data = std::unique_ptr<T>(new T[n_rows * n_cols]);
        convert_to_row_major_onto_buffer(new_data.get());
        this->data = std::move(new_data);
    }

    void convert_to_row_major_onto_buffer(T* buffer) {
        T* data_ptr = data.get();
        for (int64_t i=0; i < n_rows; i++) {
            for (int64_t j=0; j < n_cols; j++) {
                buffer[i * n_cols + j] = data_ptr[j * n_rows + i];
            }
        }
    }

    template <typename S>
    static std::unique_ptr<DenseMatrix<T>> reindex(std::unique_ptr<DenseMatrix<S>> mat) {
        auto data = std::unique_ptr<T>(new T[mat->n_entries]);
        util::convert_numeric_arrays(data.get(), mat->data.get(), mat->n_entries);
        return std::unique_ptr<DenseMatrix<T>>(
            new DenseMatrix<T>(mat->n_rows, mat->n_cols, std::move(data)));
    }

};

}  // npeff

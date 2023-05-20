#pragma once
/**
 * Default parameter classes to use for the OnlyWLearner. Custom parameter
 * classes should extend these templates.
 */
#include <vector>

#include <misc/macros.h>
#include <util/matrices.h>


namespace Nmf {
namespace OnlyW {


template<
    typename IndT,
    typename MatrixH,
    typename MatrixA = ElCsrMatrix<IndT>
>
struct LearnerParams {
    std::vector<MatrixA>& A_row_wise_partitions;
    
    int maxIters;

    float eps;
    long seed;

    // MatrixA* A;
    MatrixH* H;

    int n_partitions;

    int main_device = 0;
    int log_every_n_steps = 25;

    LearnerParams(std::vector<MatrixA>& A_row_wise_partitions) : A_row_wise_partitions(A_row_wise_partitions) {}

    long n_cols() {
        THROWSERT(n_A_cols_()== H->n_cols);
        return H->n_cols;
    }

    long n_examples() {
        return n_A_rows_();
    }

    long n_components() {
        return H->n_rows;
    }

protected:
    long n_A_rows_() {
        long n = 0;
        for (auto& mat : A_row_wise_partitions) {
            n += mat.n_rows;
        }
        return n;
    }
    long n_A_cols_() {
        long n_cols = A_row_wise_partitions[0].n_cols;
        for (auto& mat : A_row_wise_partitions) {
            THROWSERT(mat.n_cols = n_cols);
        }
        return n_cols;
    }
};


// template<
//     typename IndT,
//     typename MatrixH,
//     typename MatrixA = ElCsrMatrix<IndT>
// >
// struct LearnerParams {
//     int maxIters;

//     float eps;
//     long seed;

//     std::vector<MatrixA> A_partitions;
//     std::vector<MatrixH> H_partitions;

//     int main_device = 0;

//     int log_every_n_steps = 25;


//     int n_partitions() {
//         THROWSERT(A_partitions.size() == H_partitions.size());
//         return A_partitions.size();
//     }

//     long n_cols() {
//         long n_cols_A = _n_cols(A_partitions);
//         long n_cols_H = _n_cols(H_partitions);
//         THROWSERT(n_cols_A == n_cols_H);
//         return n_cols_A;
//     }

//     long n_examples() {
//         return A_partitions[0].n_rows;
//     }

//     long n_components() {
//         return H_partitions[0].n_rows;
//     }

// protected:
//     template<typename Matrix>
//     long _n_cols(std::vector<Matrix>& partitions) {
//         long n = 0;
//         for (auto& mat : partitions) {
//             n += mat.n_cols;
//         }
//         return n;
//     }

// };


template<
    typename IndT,
    typename MatrixH,
    typename MatrixA = ElCsrMatrix<IndT>,
    typename LearnerParams = LearnerParams<IndT, MatrixH, MatrixA>
>
struct LearnerDeviceContextParams {
    LearnerParams& learnerParams;

    int device;
    int n_devices;

    MatrixA& A;
    MatrixH& H;

    long n_rows;
    long n_cols;
    long slice_n_cols;

    LearnerDeviceContextParams(LearnerParams& learnerParams, int device) : 
        learnerParams(learnerParams),
        device(device),
        n_devices(learnerParams.n_devices()),
        A(learnerParams.A_partitions[device]),
        H(learnerParams.H_partitions[device]),
        n_rows(H.n_rows),
        n_cols(learnerParams.n_cols()),
        slice_n_cols(H.n_cols)
    {}

};



}
}

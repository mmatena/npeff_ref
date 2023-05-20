/*
 * Learns W given a given a fixed sparse H and a set of
 * sparse PEFs.
 */
#include <numeric>
#include <iostream>
#include <string>
#include <memory>

#include "util/flag_util.h"

#include "io/pef_fisher_h5.h"
#include "io/nmf_decomp_h5.h"

#include <cuda/cuda_system.h>
#include <nmf/sparse2/nmf_sparse_H.h>



template<typename IndT_> 
using Learner = Nmf::OnlyW::SparseH::Learner<IndT_>;


struct ProgramFlags {
    std::string outputPath;

    std::string pefPath;
    std::string nmfPath;

    int n_splits_sparse_matmul;
    int n_row_splits_pefs;

    int maxIters;
    int n_examples = -1;
    int n_fisherValues = -1;

    int main_device = 0;

    // float eps;
    // long seed;

    // Dev only flags.
    int DEV_n_cols = -1;

    static ProgramFlags readIn() {
        ProgramFlags flags;
        FlagUtil::Flag<std::string>("output_path").read(&flags.outputPath);
        FlagUtil::Flag<std::string>("pef_path").read(&flags.pefPath);
        FlagUtil::Flag<std::string>("H_path").read(&flags.nmfPath);

        FlagUtil::Flag<int>("n_splits_sparse_matmul").read(&flags.n_splits_sparse_matmul);
        FlagUtil::Flag<int>("n_row_splits_pefs").read(&flags.n_row_splits_pefs);

        FlagUtil::Flag<int>("nmf_max_iter").read(&flags.maxIters);

        FlagUtil::Flag<int>("n_examples", false).read(&flags.n_examples);
        FlagUtil::Flag<int>("n_fisher_values", false).read(&flags.n_fisherValues);

        FlagUtil::Flag<int>("main_device", false).read(&flags.main_device);

        // Dev only flags.
        FlagUtil::Flag<int>("DEV_n_cols", false).read(&flags.DEV_n_cols);

        FlagUtil::VerifyNoUnreadFlags();

        return flags;
    }
};


//////////////////////////////////////////////////////////////////////////////////////////////


void CheckDevices() {
    if (Cuda::GetDeviceCount() != 1) {
        THROW_MSG("Only single GPU is supported.");
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////

template<typename IndT>
Cuda::Host::CsrMatrix<int32_t> GetH(SparseNmfDecomposition<IndT>& sp_nmf) {
    return sp_nmf.H->move_to_csr_matrix().ReIndexWithInt32();
}

template<>
Cuda::Host::CsrMatrix<int32_t> GetH(SparseNmfDecomposition<int32_t>& sp_nmf) {
    return sp_nmf.H->move_to_csr_matrix();
}

//////////////////////////////////////////////////////////////////////////////////////////////

// template <typename IndT>
// struct ProcessedPefs {
//     PefSparseFishers<IndT> pef;
//     SparseNmfDecomposition<IndT> sp_nmf;

//     std::vector<Cuda::Host::CsrMatrix<int32_t>> fisher_row_splits;
//     Cuda::Host::CsrMatrix<int32_t>* H;
// };

struct ProcessedInputs {
    long full_dense_size;
    int32_t* index_to_og_index = nullptr;

    std::vector<Cuda::Host::CsrMatrix<int32_t>> fisher_row_splits;
    Cuda::Host::CsrMatrix<int32_t>* H = nullptr;
};


template <typename IndT>
ProcessedInputs ReadInAndPreprocess(ProgramFlags& flags) {
    using CsrMatrix32 = Cuda::Host::CsrMatrix<int32_t>;

    // TODO: Some checking on whether this needs int64 indices.
    std::cout << "Starting to read in H.\n";
    auto sp_nmf = SparseNmfDecomposition<IndT>::read(flags.nmfPath);
    std::cout << "H read in.\n";

    auto H = new CsrMatrix32(GetH(sp_nmf));

    std::cout << "Starting to read in PEFs.\n";
    auto pef = PefSparseFishers<IndT>::read(flags.pefPath, flags.n_examples, flags.n_fisherValues);
    std::cout << "PEFS read in.\n";

    pef.normalizeFishers_inPlace();

    Cuda::Host::CsrMatrix<IndT> fishers = pef.fishers->move_to_csr_matrix();

    fishers.RetainColumns_InPlace(
        std::vector<IndT>(sp_nmf.indexToOgIndex, sp_nmf.indexToOgIndex + H->n_cols));


    // The following block of code is just for development purposes.
    if (flags.DEV_n_cols >= 0) {
        std::cout << "NOTE: Using flag intended only for development purposes.\n";

        std::vector<IndT> v(flags.DEV_n_cols);
        std::iota(std::begin(v), std::end(v), 0);
        fishers.RetainColumns_InPlace(v);

        std::vector<int32_t> v32(flags.DEV_n_cols);
        std::iota(std::begin(v32), std::end(v32), 0);
        H->RetainColumns_InPlace(v32);
    }


    auto fisher_row_splits_ = fishers.SplitRowWise(flags.n_row_splits_pefs);

    ProcessedInputs ret;
    ret.H = H;

    ret.index_to_og_index = new int32_t[H->n_cols];
    Misc::ConvertNumericArrays(ret.index_to_og_index, sp_nmf.indexToOgIndex, H->n_cols);

    ret.full_dense_size = sp_nmf.fullDenseSize;

    for(auto& split : fisher_row_splits_) {
        if (!split.CanUseInt32Indices()) {
            std::cout << "Row split of PEFs cannot use int32 indices [nnz = " << split.nnz << "].\n";
            THROW;
        }
        ret.fisher_row_splits.emplace_back(std::move(split.ReIndexWithInt32()));
    }

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////////


template <typename IndT>
int maine_(ProgramFlags& flags) {
    ProcessedInputs inputs = ReadInAndPreprocess<IndT>(flags);
    auto& H = inputs.H;

    // Run as int32_t.
    typename Learner<int32_t>::Params p(inputs.fisher_row_splits);
    p.H = H;
    p.n_partitions = flags.n_splits_sparse_matmul;
    p.maxIters = flags.maxIters;
    p.main_device = flags.main_device;
    p.eps = 1e-7;
    // p.seed = 

    Learner<int32_t> learner(p);
    learner.Run();

    // Save to file.

    auto W_ = learner.GetOnHost_W();
    auto W = MeMatrix::MoveFromDenseMatrix(W_);
    W.toRowMajor_inPlace();

    auto H2 = ElCsrMatrix<int32_t>::MoveFromSparseMatrix(*H);

    SparseNmfDecomposition<int32_t> out_nmf;
    out_nmf.W = &W;
    out_nmf.H = &H2;
    out_nmf.indexToOgIndex = inputs.index_to_og_index;
    out_nmf.fullDenseSize = inputs.full_dense_size;

    out_nmf.save(flags.outputPath);

    return 0;
}


int main(int argc, char *argv[]) {
    FlagUtil::InitializeGlobalState(argc, argv);
    ProgramFlags flags = ProgramFlags::readIn();

    CheckDevices();

    // TODO: Also take H into account. Right now, I am assuming that it can always be represented
    // using int32_t indices.
    bool use_int64 = Pef::pefRequiresInt64Indices(flags.pefPath, flags.n_examples, flags.n_fisherValues);

    if(use_int64) {
        return maine_<int64_t>(flags);
    } else {
        return maine_<int32_t>(flags);
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////



template <typename IndT>
int main_(int argc, char *argv[]) {
    FlagUtil::InitializeGlobalState(argc, argv);

    ProgramFlags flags = ProgramFlags::readIn();

    if (
        !std::is_same<IndT, int64_t>::value
        && Pef::pefRequiresInt64Indices(flags.pefPath, flags.n_examples, flags.n_fisherValues)
    ) {
        std::cout << "Due to sparse-sparse multiplication stuff, not currently supporting int64 indices.\n";
        THROW;
    }

    std::cout << "Starting to read in PEFs.\n";
    PefSparseFishers<IndT> pef = PefSparseFishers<IndT>::read(flags.pefPath, flags.n_examples, flags.n_fisherValues);
    std::cout << "PEFS read in.\n";

    pef.normalizeFishers_inPlace();

    // TODO: Some checking on whether this needs int64 indices.
    std::cout << "Starting to read in H.\n";
    SparseNmfDecomposition<IndT> sp_nmf = SparseNmfDecomposition<IndT>::read(flags.nmfPath);
    std::cout << "H read in.\n";


    Cuda::Host::CsrMatrix<IndT> fishers = pef.fishers->move_to_csr_matrix();
    Cuda::Host::CsrMatrix<IndT> H = sp_nmf.H->move_to_csr_matrix();

    fishers.RetainColumns_InPlace(
        std::vector<IndT>(sp_nmf.indexToOgIndex, sp_nmf.indexToOgIndex + H.n_cols));

    // The following block of code is just for development purposes.
    if (flags.DEV_n_cols >= 0) {
        std::cout << "NOTE: Using flag intended only for development purposes.\n";
        std::vector<IndT> v(flags.DEV_n_cols);
        std::iota(std::begin(v), std::end(v), 0);
        fishers.RetainColumns_InPlace(v);
        H.RetainColumns_InPlace(v);
    }

    int n_devices = Cuda::GetDeviceCount();
    // THROWSERT(flags.main_device < n_devices);
    if (n_devices != 1) {
        std::cout << "Only single GPU is supported.\n";
        THROW;
    }

    auto fisher_row_splits = fishers.SplitRowWise(flags.n_row_splits_pefs);

    typename Learner<IndT>::Params p(fisher_row_splits);
    p.H = &H;
    p.n_partitions = flags.n_splits_sparse_matmul;
    p.maxIters = flags.maxIters;
    p.main_device = flags.main_device;
    p.eps = 1e-7;
    // p.seed = 

    Learner<IndT> learner(p);
    learner.Run();

    // Save to file.

    auto W_ = learner.GetOnHost_W();
    auto W = MeMatrix::MoveFromDenseMatrix(W_);
    W.toRowMajor_inPlace();

    auto H2 = ElCsrMatrix<IndT>::MoveFromSparseMatrix(H);

    SparseNmfDecomposition<IndT> out_nmf;
    out_nmf.W = &W;
    out_nmf.H = &H2;
    out_nmf.indexToOgIndex = sp_nmf.indexToOgIndex;
    out_nmf.fullDenseSize = sp_nmf.fullDenseSize;

    out_nmf.save(flags.outputPath);

    return 0;
}

// int main(int argc, char *argv[]) {
//     // TODO: Choose depending on size of pefs and maybe H.

//     // return main_<int32_t>(argc, argv);
//     return main_<int64_t>(argc, argv);
// }

// Transfer and split H to device as I have it now.
// Split A by rows on host. (make each representable using int32 indices?)
// Transfer A split by split to device, computing and then clearing memory
// each time.


/*
General TODOS:
 - write more tests, use actualy framework (and build framework too)
 - do the sp-sp matmuls one at a time.
 - allow for "backing off / increasing" memory for the second buffer of sp-sp MM.
 - Maybe allow sp-sp matmuls to be split up.


 - Add early stopping condition based on changes in the loss.

 - Handle partitions of the spsp matmul where at least one factor has nnz=0.


- Maybe re-run some NMF decomps with the fixed sparsity thing.
*/

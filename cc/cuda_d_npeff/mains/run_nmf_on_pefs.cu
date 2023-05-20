
#include <iostream>
#include <string>

#include "util/cuda_system.h"
#include "util/flag_util.h"
#include "util/matrices.h"
#include "nmf/sparse/multi_mu_dense_factors_nmf1.h"
#include "util/sparse_util.h"
#include "io/pef_fisher_h5.h"
#include "io/nmf_decomp_h5.h"


const std::string FLAG_NOT_FOUND = FlagUtil::FLAG_NOT_FOUND;


struct ProgramFlags {
    std::string outputPath;
    std::string pefPath;
    int n_components;
    int maxIters;

    int n_examples = -1;
    int n_fisherValues = -1;

    // TODO: Check that, if set, this value is contained in the open interval (0, 1).
    float sparsity_W = -1.0f;

    // Inclusive, the minimum number of non-zero PEF values for a given
    // parameter to be included.
    int min_valuesPerParameter = 1;

    // Optional path to saved NMF/SparseNMF file containing the initial H. H is
    // dense and randomly initialized otherwise.
    std::string initialHPath = FLAG_NOT_FOUND;
    // Same as initialHPath but for W. A difference is that W will be dense always.
    std::string initialWPath = FLAG_NOT_FOUND;

    bool hasInitialHPath() {
        return initialHPath != FLAG_NOT_FOUND;
    }

    bool hasInitialWPath() {
        return initialWPath != FLAG_NOT_FOUND;
    }
};


ProgramFlags parseFlags(int argc, char *argv[]) {
    ProgramFlags flags;

    FlagUtil::Flag<std::string>("output_path").read(&flags.outputPath);
    FlagUtil::Flag<std::string>("per_example_fishers").read(&flags.pefPath);

    FlagUtil::Flag<int>("nmf_n_components").read(&flags.n_components);
    FlagUtil::Flag<int>("nmf_max_iter").read(&flags.maxIters);

    FlagUtil::Flag<int>("n_examples", false).read(&flags.n_examples);
    FlagUtil::Flag<int>("n_fisher_values", false).read(&flags.n_fisherValues);

    FlagUtil::Flag<int>("min_values_per_parameter", false).read(&flags.min_valuesPerParameter);

    // Not fully supported yet flags.
    FlagUtil::Flag<float>("sparsity_W", false).read(&flags.sparsity_W);

    FlagUtil::Flag<std::string>("initial_H_path", false).read(&flags.initialHPath);
    FlagUtil::Flag<std::string>("initial_W_path", false).read(&flags.initialWPath);

    return flags;
}


template <typename IndT>
struct PreprocessedPef {
    long n_cols;
    long n_rows;

    int nDevices;
    ElCsrMatrix<IndT>* fisherPartitions = nullptr;
    IndT* newToOgIndex = nullptr;
    long fisherDenseSize;

    MeMatrix* initialDenseH = nullptr;
    // MeMatrix* initialDenseW = nullptr;

    ~PreprocessedPef() {
        free(fisherPartitions);
        delete[] newToOgIndex;
    }

    bool canReindexWithInt32() {
        for(int i=0;i<nDevices;i++) {
            if(!fisherPartitions[i].canReindexWithInt32()) {
                return false;
            }
        }
        return true;
    }

    PreprocessedPef<int32_t> reindexWithInt32() {
        PreprocessedPef<int32_t> ret;
        ret.n_cols = n_cols;
        ret.n_rows = n_rows;
        ret.nDevices = nDevices;
        ret.fisherDenseSize = fisherDenseSize;
        ret.initialDenseH = initialDenseH;
        // ret.initialDenseW = initialDenseW;

        ret.newToOgIndex = new int32_t[n_cols];
        convertNumericArrays(ret.newToOgIndex, newToOgIndex, n_cols);

        ret.fisherPartitions = (ElCsrMatrix<int32_t>*) malloc(sizeof(ElCsrMatrix<int32_t>) * nDevices);
        for(int i=0;i<nDevices;i++) {
            ret.fisherPartitions[i] = fisherPartitions[i].reindexWithInt32();
        }

        return ret;
    }
};


template <typename IndT>
PreprocessedPef<IndT> readAndPreprocessPef(ProgramFlags& flags) {
    PreprocessedPef<IndT> ret;

    cudaGetDeviceCount(&ret.nDevices);




    // if (flags.hasInitialWPath()) {
    //     std::cout << "TODO: Support initial W.\n";
    //     THROW;
    // }

    // // TODO: This stuff might leak memory.
    // // TODO: Validate things like the size of H somewhere.

    // std::vector<IndT> forceRetainColumns;
    // if(flags.hasInitialHPath()) {
    //     // TODO: Make sure that the IndT will be fine.
    //     auto nmfInitH = NmfDecomposition<IndT>::read(flags.initialHPath);
    //     // std::cout << nmfInitH.fullDenseSize << "\n";
    //     forceRetainColumns = std::vector<IndT>(nmfInitH.indexToOgIndex, nmfInitH.indexToOgIndex + nmfInitH.H->n_cols);
    //     ret.initialDenseH = nmfInitH.H;


    //     // NOTE: I could potententially accomplish this when I read from host to device memory with multiple
    //     // transfers and being careful.
    //     std::cout << "TODO: THIS IS WRONG IN GENERAL. NEED TO HANDLE POTENTIAL ADDITIONAL COLUMNS TO H FROM BELOW.\n";


    //     // TODO: Make this memory management cleaner.
    //     delete nmfInitH.W;
    //     delete[] nmfInitH.indexToOgIndex;
    // }




    std::cout << "Starting to read in PEFs.\n";
    PefSparseFishers<IndT> pef = PefSparseFishers<IndT>::read(flags.pefPath, flags.n_examples, flags.n_fisherValues);
    std::cout << "PEFS read in.\n";

    pef.normalizeFishers_inPlace();
 
    std::cout << "n_cols before reduction: " << pef.fishers->n_cols << "\n";
    IndT* newToOgIndex;
    // removeSmallestL0NormColumns_inPlace(pef.fishers, &newToOgIndex, flags.min_valuesPerParameter, forceRetainColumns);
    removeSmallestL0NormColumns_inPlace(pef.fishers, &newToOgIndex, flags.min_valuesPerParameter);
    std::cout << "n_cols after reduction: " << pef.fishers->n_cols << " [density=" << pef.fishers->density() << "]\n";

    ret.newToOgIndex = newToOgIndex;
    ret.fisherDenseSize = pef.fisherDenseSize;

    ret.n_rows = pef.fishers->n_rows;
    ret.n_cols = pef.fishers->n_cols;

    ret.fisherPartitions = splitColumnWise(*pef.fishers, ret.nDevices);

    return ret;
}


template <typename IndT>
int _run(ProgramFlags& flags, PreprocessedPef<IndT>& p_pef) {
    IndT* newToOgIndex = p_pef.newToOgIndex;

    // TODO: Set these somewhere else and/or flags and/or let seed to change.
    long seed = 43190432042;
    // float eps = 1e-12;
    float eps = 1e-8;

    std::cout << "Initializing NMF.\n";
    MuNmfParams p;
    p.rank = flags.n_components;
    p.max_iters = flags.maxIters;
    p.eps = eps;
    p.seed = seed;
    p.sparsenessConstraints_W.l1_norm = flags.sparsity_W;

    MuNmf<IndT> nmf(p_pef.fisherPartitions, p);

    std::cout << "Running NMF.\n";
    nmf.run();

    MeMatrix W = nmf.loadWToHostSync();
    W.toRowMajor_inPlace();

    MeMatrix H = nmf.loadHToHostSync();
    H.toRowMajor_inPlace();

    NmfDecomposition<IndT> decomp;
    decomp.W = &W;
    decomp.H = &H;

    decomp.indexToOgIndex = newToOgIndex;
    decomp.fullDenseSize = p_pef.fisherDenseSize;

    std::cout <<"Saving decomposition to file.\n";
    decomp.save(flags.outputPath);

    return 0;
}


int main(int argc, char *argv[]) {
    FlagUtil::InitializeGlobalState(argc, argv);

    // auto nmf = NmfDecomposition<int32_t>::read("/tmp/asdf.h5");

    std::cout << "THERE STILL ARE BUGS!!! COMPARE MANGO (GOOD) TO BANANA (BAD)\n";

    ProgramFlags flags = parseFlags(argc, argv);

    if (Pef::pefRequiresInt64Indices(flags.pefPath, flags.n_examples, flags.n_fisherValues)) {
        std::cout << "Reading in PEFs using 64 bit indices.\n";
        PreprocessedPef<int64_t> p_pef = readAndPreprocessPef<int64_t>(flags);
        if (p_pef.canReindexWithInt32()) {
            std::cout << "Running NMF using 32 bit indices.\n";
            PreprocessedPef<int32_t> p_pef2 =  p_pef.reindexWithInt32();
            // TODO: Free memory associated with p_pef before starting the NMF.
            return _run<int32_t>(flags, p_pef2);
        } else {
            std::cout << "Running NMF using 64 bit indices.\n";
            return _run<int64_t>(flags, p_pef);
        }
    } else {
        std::cout << "Using 32 bit indices.\n";
        PreprocessedPef<int32_t> p_pef = readAndPreprocessPef<int32_t>(flags);
        std::cout << "Running NMF using 32 bit indices.\n";
        return _run<int32_t>(flags, p_pef);
    }

}



/*
- Separate the loading and A-splitting from the other steps. This is to allow us to use int32_t if possible.

*/

// int main(int argc, char *argv[]) {
//     // std::cout << "TODO: Probable errors.\n";
//     // // write tests (e.g. for loss), probably make a computeLossUncached function.
//     // // Also add basic C++/cuda tooling to sublime text.

//     // // Add safety checks for int32/int64 to prevent overflow.
//     // return exit(0);

//     std::cout << "THERE STILL ARE BUGS!!! COMPARE MANGO (GOOD) TO BANANA (BAD)\n";

//     // std::cout << "maxOutputMag: " << maxOutputMag << "\n";
//     ProgramFlags flags = parseFlags(argc, argv);

//     if (Pef::pefRequiresInt64Indices(flags.pefPath, flags.n_examples, flags.n_fisherValues)) {
//         // TODO: Maybe try splitting up the matrices and see if that is faster than 64 bit indices.
//         std::cout << "Using 64 bit indices.\n";
//         return _main<int64_t>(flags);
//     } else {
//         std::cout << "Using 32 bit indices.\n";
//         return _main<int32_t>(flags);
//     }

// }


// Support cases where the number of columns is not exactly divisible by
// the number of devices.
// 
// Support starting from a saved NMF file (will still need pef to be passed).
// 
// Add command-line support for some more of the other parameters.
// 
// Compute fishers for both HANS train and test sets together.
// 
// Add support for sparsifying H when saving with some value cut-off. Explore
// the dense Hs to get idea for value cut-off. Or maybe put that in some other
// executable. Or allow to save both.




// ln -s /fruitbasket/users/mmatena/.virtualenvs /home/mmatena/.virtualenvs
// ln -s /fruitbasket/users/mmatena/.cache /home/mmatena/.cache

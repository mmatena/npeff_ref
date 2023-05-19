// #pragma once
// // Stuff related to initialization of the W and G.

// #include <cstdint>
// #include <string>

// #include "./config.h"

// namespace npeff {
// namespace factorization {


// struct InitializationConfig {
//     enum ColumnPruningPolicy {
//         REQUIRE_EXACT_MATCH,
//         FROM_INITIALIZATION,
//         // INTERSECTION,
//     };

//     std::string initialization_filepath;
//     bool use_W;
//     ColumnPruningPolicy column_pruning_policy;

//     // See whether we are using an initialization from a file.
//     bool has_initialization() const {
//         return initialization_filepath.empty();
//     }
// };


// // Checks to see if there are any immediate incompatabilities. Note that
// // this function could return true but still have an incompatable initialization.
// bool check_immediate_compatability(const FactorizationConfig& config, const InitializationConfig& init_config) {
//     // File exists.
//     // Rank is same.
    
// }

// // ensure_exists_and_is_consistent_with_flags
//     // rank and whatnot
// // Need to read

// // Different column_pruning? -> intersection??? or force from initialization

// // FLAGS initialization_column_pruning_policy: "from_initialization", "require_exact_match", "intersection"



// }  // factorization
// }  // npeff

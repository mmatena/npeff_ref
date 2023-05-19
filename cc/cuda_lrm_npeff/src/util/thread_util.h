#pragma once
// Utilities related to threads.

#include <memory>
#include <thread>
#include <vector>

namespace npeff {
namespace util {


void join_threads(std::vector<std::thread>& threads) {
    for (auto& thread : threads) { thread.join(); }
}


}  // util
}  // npeff

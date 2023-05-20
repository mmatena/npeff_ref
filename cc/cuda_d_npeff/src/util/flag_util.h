#pragma once

#include <vector>
#include <unordered_set>
#include <iostream>
#include <string>

#include <util/cuda_statuses.h>


namespace FlagUtil {


const std::string FLAG_NOT_FOUND = "__FLAG_NOT_FOUND__";


// Please forgive me for the sin that I am about to commit.
namespace _GlobalState {
    int argc;
    char **argv;
    std::unordered_set<std::string> read_flags;
};

void InitializeGlobalState(int argc, char *argv[]) {
    _GlobalState::argc = argc;
    _GlobalState::argv = argv;
}


std::string _getFlagAsStr(int argc, char *argv[], const std::string& name, bool required = true) {
    std::string prefix = "--" + name + "=";

    for (int i=0; i < argc; i++) {
        std::string flag(argv[i]);
        if (flag.rfind(prefix, 0) == 0) {
            return flag.substr(prefix.size());
        }
        
    }

    if (required) {
        std::cout << "Missing flag: " << name << "\n";
        throw;
    }

    return FLAG_NOT_FOUND;
}


// namespace internal {
// } // internal

bool AreUnreadFlagsPresent() {
    // Starting at 1 since the first argv will be the name of the program.

    // std::vector<std::string> unreads;

    // for (int i=1; i < _GlobalState::argc; i++) {
    //     std::string flag(_GlobalState::argv[i]);



    // }
    return _GlobalState::read_flags.size() != _GlobalState::argc - 1;
}


void VerifyNoUnreadFlags() {
    if (AreUnreadFlagsPresent()) {
        std::cout << "Unrecognized flags.\n";
        throw;
    }
}


template<typename T>
class Flag {
public:
    // This code recomputes stuff a bunch of times, but the computations are cheap
    // and few, so it doesn't really matter.
    const std::string name;
    const bool required;

    Flag(const std::string& name, bool required = true) : name(name), required(required) {}
    Flag(const std::string name, bool required = true) : name(name), required(required) {}
    Flag(std::string& name, bool required = true) : name(name), required(required) {}
    Flag(const char name[], bool required = true) : name(name), required(required) {}

    bool isPresent() { return _getFlagStr() != FLAG_NOT_FOUND; }

    T value() {
        if(!isPresent()) {
            std::cout << "Attempted to get value of a flag that was not found.\n";
            THROW;
        }
        return _parse(_getFlagStr());
    }

    T value(T defaultValue) {
        if (required) {
            std::cout << "Default flag values not allowed for required flags.\n";
            THROW;
        }
        if(!isPresent()) {
            return defaultValue;
        }
        return _parse(_getFlagStr());
    }

    void read(T* writeLocation) {
        if (!isPresent() && !required) { return; }
        _GlobalState::read_flags.insert(name);
        *writeLocation = value();
    }

private:
    T _parse(std::string flagStr);

    std::string _getFlagStr() {
        return _getFlagAsStr(_GlobalState::argc, _GlobalState::argv, this->name, required);
    }
};


template<>
std::string Flag<std::string>::_parse(std::string flagStr) {
    return flagStr;
}

template<>
int Flag<int>::_parse(std::string flagStr) {
    return std::stoi(flagStr);
}

template<>
long Flag<long>::_parse(std::string flagStr) {
    return std::stol(flagStr);
}

template<>
float Flag<float>::_parse(std::string flagStr) {
    return std::stof(flagStr);
}





}

#pragma once

namespace Cuda {
namespace Transferable {



template<typename HostT, typename DeviceT>
class Pair {
public:
    HostT& host;
    DeviceT& device;

    Pair(HostT& host, DeviceT& device) :
        host(host),
        device(device)
    {}
    
};



}
}

#include "HipDeviceGuard.h"

namespace llaminar2
{

    thread_local int HipDeviceGuard::current_device_ = -1;

} // namespace llaminar2

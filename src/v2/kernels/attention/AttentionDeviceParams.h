#pragma once

namespace llaminar2
{
    namespace attention
    {
        struct AttentionDeviceParams
        {
            int kv_len = 0;
            int position_offset = 0;
            int mask_stride = 0;
        };
    } // namespace attention
} // namespace llaminar2

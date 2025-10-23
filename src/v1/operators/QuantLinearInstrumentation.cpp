#include "QuantLinearInstrumentation.h"
namespace llaminar
{
    QuantLinearInstrumentation &quantLinearInstr()
    {
        static QuantLinearInstrumentation inst;
        return inst;
    }
}

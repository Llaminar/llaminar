/**
 * @file Test__RCCLConfiguredLibrary.cpp
 * @brief Bind the shared inert replay to the complete production RCCL loader.
 * No SDK, runtime or device is loaded. Native optional-symbol behavior remains
 * covered by the same source-body fixture and exact legacy observation mode.
 */
#define LLAMINAR_RCCL_PATH_REPLAY 1
#define NCCLConfiguredLibrary RCCLConfiguredLibrary
#include "Test__NCCLConfiguredLibrary.cpp"

/**
 * @file NativeClosureReplayModule.cpp
 * @brief Unrelated tiny loadable module for original loader-snapshot concurrency replay.
 */
/** @return Fixed independent payload proving the loader actually opened this module. */
extern "C" int llaminarNativeClosureReplayValue(){return 17;}

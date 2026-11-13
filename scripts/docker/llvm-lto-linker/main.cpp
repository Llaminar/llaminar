/** @file main.cpp
 * @brief Private exact-source lld entry; no installed tool or runtime is modified.
 */
#include "llvm/Support/LLVMDriver.h"
int lld_main(int, char **, const llvm::ToolContext &);
/**
 * @brief Dispatch directly to the exact pinned lld driver implementation.
 * @param argc Original process argument count, including the executable path.
 * @param argv Original linker arguments; no policy or target flags are rewritten.
 * @return Native linker exit code, including any failed allocation-record setup.
 */
int main(int argc, char **argv) {
  return lld_main(argc, argv, {argv[0], nullptr, false});
}

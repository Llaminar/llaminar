/** @file GVNRemarkProbe.cpp
 * @brief Exercise the actual archive-owned GVN diagnostic guard without GPU work.
 *
 * Each fresh context runs the actual GVN pass on one clobbered-load IR fixture.
 * The ordinary LLVM diagnostic callback observes remark construction even when
 * serialization filters it out. Its enable predicates remain explicit, so the
 * observer does not itself request the expensive GVN diagnostic. The two linked
 * executables differ only in the original/repaired SDK archive members.
 */
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

/** @brief Observe constructed GVN remarks while independently owning terminal selection. */
class Observer final : public DiagnosticHandler {
public:
  bool TerminalGVN;
  std::vector<std::string> Messages;

  /** @brief Select requested terminal remarks without enabling observation implicitly.
   * @param terminalGVN Whether the terminal diagnostic policy explicitly enables GVN.
   */
  explicit Observer(bool terminalGVN) : TerminalGVN(terminalGVN) {}

  /** @brief Record actual LoadClobbered construction through LLVM's normal callback.
   * @param info Original diagnostic delivered by the actual GVN pass.
   * @return True after consuming the diagnostic; nothing is printed or synthesized.
   */
  bool handleDiagnostics(const DiagnosticInfo &info) override {
    if (const auto *remark = dyn_cast<DiagnosticInfoOptimizationBase>(&info)) {
      if (remark->getPassName() == "gvn" && remark->getRemarkName() == "LoadClobbered") {
        std::string message;
        raw_string_ostream output(message);
        DiagnosticPrinterRawOStream printer(output);
        info.print(printer);
        output.flush();
        Messages.push_back(message);
      }
    }
    return true;
  }

  /** @brief Keep missed-remark selection independent of the observing callback.
   * @param pass Original pass name queried by the optimization emitter.
   * @return True only for an explicitly requested terminal GVN missed remark.
   */
  bool isMissedOptRemarkEnabled(StringRef pass) const override {
    return TerminalGVN && pass == "gvn";
  }

  /** @brief Exclude unrelated passed remarks from this fixed diagnostic contract.
   * @param pass Original queried pass name.
   * @return False; this witness never requests passed remarks.
   */
  bool isPassedOptRemarkEnabled(StringRef pass) const override { return false; }

  /** @brief Exclude unrelated analysis remarks from terminal selection.
   * @param pass Original queried pass name.
   * @return False; saved stack-frame evidence belongs to the separate native proof.
   */
  bool isAnalysisRemarkEnabled(StringRef pass) const override { return false; }
};

/** @brief Author one fixed remark-selection case without compiler command-line overrides. */
struct Case {
  const char *Name;
  bool StreamEnabled;
  const char *Filter;
  bool TerminalGVN;
};

/** @brief Execute fresh actual GVN state and preserve decisions plus diagnostic observations.
 * @param item Fixed stream/filter/terminal contract for this invocation.
 * @return Observed case, or an Error if LLVM rejects the original IR or remark setup.
 */
static Expected<json::Object> observe(const Case &item) {
  LLVMContext context;
  context.setDiagnosticsHotnessThreshold(0);
  auto handler = std::make_unique<Observer>(item.TerminalGVN);
  Observer *observer = handler.get();
  // This observer sees construction without silently enabling terminal remarks.
  context.setDiagnosticHandler(std::move(handler), false);
  std::string serialized;
  raw_string_ostream remarks(serialized);
  if (item.StreamEnabled) {
    if (Error error = setupLLVMOptimizationRemarks(context, remarks, item.Filter, "yaml", false))
      return std::move(error);
  }
  SMDiagnostic diagnostic;
  auto module = parseAssemblyString(R"IR(
    define i32 @witness(ptr %p, ptr %q) {
    entry:
      %before = load i32, ptr %p
      store i32 1, ptr %q
      %after = load i32, ptr %p
      %sum = add i32 %before, %after
      ret i32 %sum
    }
  )IR", diagnostic, context);
  if (!module) {
    if (item.StreamEnabled)
      finalizeLLVMOptimizationRemarks(context);
    return createStringError(inconvertibleErrorCode(), "Original GVN witness IR was rejected");
  }
  LoopAnalysisManager loops;
  FunctionAnalysisManager functions;
  CGSCCAnalysisManager cgscc;
  ModuleAnalysisManager modules;
  PassBuilder builder;
  builder.registerModuleAnalyses(modules);
  builder.registerCGSCCAnalyses(cgscc);
  builder.registerFunctionAnalyses(functions);
  builder.registerLoopAnalyses(loops);
  builder.crossRegisterProxies(loops, functions, cgscc, modules);
  FunctionPassManager passes;
  passes.addPass(GVNPass());
  passes.run(*module->getFunction("witness"), functions);
  if (item.StreamEnabled)
    finalizeLLVMOptimizationRemarks(context);
  std::string optimizedIR;
  raw_string_ostream output(optimizedIR);
  module->print(output, nullptr);
  output.flush();
  remarks.flush();
  json::Array messages;
  for (const std::string &message : observer->Messages)
    messages.push_back(message);
  return json::Object{{"name", item.Name}, {"stream_enabled", item.StreamEnabled},
      {"filter", item.Filter}, {"terminal_gvn", item.TerminalGVN},
      {"constructed", static_cast<int64_t>(observer->Messages.size())},
      {"messages", std::move(messages)}, {"serialized", serialized}, {"optimized_ir", optimizedIR}};
}

/** @brief Run the actual diagnostic selector matrix; no target/runtime/device is initialized.
 * @return Zero after publishing complete observations, nonzero on original LLVM failure.
 */
int main() {
  const Case cases[] = {{"filtered", true, "stack-frame-layout", false},
      {"gvn_selected", true, "gvn", false}, {"unfiltered", true, "", false},
      {"terminal_override", true, "stack-frame-layout", true},
      {"terminal_only", false, "", true}, {"disabled", false, "", false}};
  json::Array observations;
  for (const Case &item : cases) {
    auto result = observe(item);
    if (!result) {
      logAllUnhandledErrors(result.takeError(), errs(), "GVN witness: ");
      return 1;
    }
    observations.push_back(std::move(*result));
  }
  outs() << formatv("{0:2}\n", json::Value(json::Object{{"schema", 1}, {"cases", std::move(observations)}}));
  return 0;
}

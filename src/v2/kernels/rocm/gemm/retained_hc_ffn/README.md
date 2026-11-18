# Experimental retained QFN M1 HC/FFN

`LLAMINAR_QFN_RETAINED_M1_HC_FFN_CANDIDATE=1` selects the original QFN
main-inference decode graph with one request and one row for experimental
lowering. The participant owner admits the exact gfx906 60-CU implementation,
original FP16 router, H2560/R320/E512/topK10 and exact local routed/shared
widths from the ordinary QFN TP1, TP2, TP4 and TP8 source assignments. It also
admits the F256/S192 matched reference fixture. The HC and shared projections
use native Q8_0. Routed projections use the source-authenticated
Q4_K/Q4_K/Q5_1, Q4_K/Q4_K/Q8_0 or Q5_K/Q5_K/Q8_0 tuple. A shared
device-owned row count defines active and empty replay. The
original graph declares the collective and any following standalone FP32 injection.
The retained program executes only the participant-local previous write, HC
read, router and routed/shared FFN. The opt-in defaults to off.

The graph's cold output-demand seal resolves canonical tensor consumers and
selected snapshots. An outside selected-descriptor reader must declare its exact
producer, device-local runtime owner, and model layer and depend directly on
that producer. The seal and admitted native function are capture identities.
The closed complete suffix reads its acquired original bank directly. An
ordered outside reader retains the original selected gate, up and down table
publications. Both variants publish original IDs, route weights, local route
records and demand counters. The closed entry with no optional HC/FFN outputs
binds the original selected and local histogram writer together. The external
entry publishes its three selected descriptor records before the retained node
retires. The closed entry has no outside descriptor stores. With no optional HC/FFN
outputs it also pairs eight-lane groups within each wave for the HC-up fold.
The two partners consume consecutive blocks and pass the first root to the
second. The native ADDs preserve the original accumulator and final primary
operand priorities, including exceptional floating-point values.

The original workspace manager owns the borrowed descriptor slots and
exclusive retained scratch lease. Preparation retains the original tables in
both variants. Physical admission, accounting, and retirement remain with
the original arena authorities. Expert movement and publication stay with the
original placement store, transfer events, and epoch boundaries. The retained
owner neither acquires a second placement authority nor changes the lifetime
of a bank. All-GPU movement copies native projection planes. A family with a
CPU participant also admits the CPU-normalized execution format, even when
that participant initially has no experts.

`V2_Integration_ROCmRetainedM1DescriptorDemandHostABI` runs the actual
HIP-compiled selector from the ordinary CXX compiler without opening a GPU.
The release test owner can be explicitly selected with
`LLAMINAR_RELEASE_NATIVE_TEST_TARGETS=v2_integration_retained_m1_descriptor_demand_host_abi`.
`V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm` replays the native
closed writer against the original staged numerical and runtime ledger oracle
on one discovered rank-local gfx906 device. It uses generated native sources.
Native movement, grouped verification, and serving economy have independent gates.

Set `LLAMINAR_QFN_RETAINED_TP_GGUF` at configure time to the original local
GGUF or first shard of a split GGUF. The optional
`V2_Integration_QFNRetainedTP_OriginalSource` CTest verifies native source
windows for the four canonical equal-weight layouts on the host.
`V2_Integration_QFNRetainedTP{1,2,4,8}_LocalSourceWindow_ROCm` uses the same
source planner and compares a selected layer and member against the original
staged capture on the GPU named by `LLAMINAR_TEST_RETAINED_HC_FFN_PCI`.
`LLAMINAR_TEST_QFN_SOURCE_LAYER` and `LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT`
choose one original source window. Each test exercises a single local
participant. Distributed TP and grouped MTP have their own collective,
publication and numerical obligations.

The closed gfx906 leaf also uses the packed register-queue selector. Its
ordered FP32 key and expert ID travel together through the native root. The
winning lane reloads the original value to preserve signed-zero and NaN
payload bytes. This path uses the original lane-zero normalization order.
The HC norm, paired HC activation and FFN quantizers use the shared bounded
positive binary16 publication primitive. This entry's floating-point mode
preserves half denormals and rounds to nearest, including the minimum scale.
External descriptor consumers and canonical observers use the existing
selector and descriptor-independent scale publication.

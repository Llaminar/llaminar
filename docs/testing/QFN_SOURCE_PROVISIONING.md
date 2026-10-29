# QFN source-admission provisioning

Source-admission discovery owns test prerequisites separately from production
placement and PMA admission. Schema 2 source metadata and schema 2 native hardware
metadata are required together; rebuild discovery instead of accepting schema 1
records that cannot carry physical VRAM prerequisites.

Only original CUDA TP2 fixed-head VRAM, eight-GPU-expert, active/full15 positive
source cases require at least 32GiB of actual total physical memory on each
participating CUDA GPU. This conservative author-owned larger-card test class
covers 24 registrations: two communicator orders, four movement/owner policies,
and three active MTP policies. It is neither a current-free-memory threshold nor
a calculated model budget. Marketing capacity and an unrelated larger GPU do
not satisfy an undersized participating device.

Hosts below this prerequisite report those exact cases as
`hardware-capacity-ineligible-unqualified`, retaining observed and required
bytes for every undersized participant. This is missing larger-card coverage,
not a successful admission or proof that the requested topology is unsupported.
Qualifying hosts must execute the original positive contract; a PMA or runtime
failure remains red. Missing or malformed required physical observations fail
metadata validation rather than counting as zero capacity.

Off controls, explicit four-expert retained3 positives, and every rejection
control remain mandatory when their device membership is available. The public
13GiB-cap negative remains a distinct capacity-rejection obligation. Numerical
model/reference policy, CTest identities, production placement, and all retained
MTP defaults remain unchanged. PMA remains the sole physical allocation ledger.

The separate CUDA2→CPU2 pipeline family with 15 GPU layers and 33 CPU layers
retains the original full15 MTP capacity and every expert/storage policy. It
has its own identities and does not replace the original 16/32-layer positives.
These pipeline families have no inferred larger-card prerequisite: selected
cases must prove actual fit through PMA, and a capacity failure remains red.

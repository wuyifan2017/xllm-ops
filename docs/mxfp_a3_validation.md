# A5 MTP Optimization And A3 Regression

This branch starts from upstream xllm-ops
`68033814eea35b96addec2f8cef12785accb364f` (2026-10-14 snapshot).
Both `mega_gdn_mtp_decode_pto_kernel.h` headers receive A5 conv/norm barrier
and conversion optimizations with an independent final-output event. Upstream
host/tiling ABI and kernel signatures are retained.

Non-A5 operations are preserved. Preparation checks compare the complete
preprocessed non-A5 token streams, expanding the new helper wrappers and event
alias, against upstream in vector and cube modes for both headers.
This check does not execute an A3 kernel.
All four generic/arch35 vector/cube comparisons passed in the isolated
preparation environment. Logs are retained in
`/data/w30082966/mxfp_upstream_merge_20261014/validation/port` in `wyf_A5_Perf`.

For A3 acceptance, build this revision and the named upstream control with the
same real A3 toolchain. Use private install roots and assigned idle devices.
Run existing MTP numerical tests with checkpoint/accepted-token feedback and
graph replay, then the associated xLLM BF16/INT8 generation regressions.
Record source/binary/OPP hashes, completed/total counts and raw outputs.
No A3 hardware run or new performance claim is part of this source delivery.

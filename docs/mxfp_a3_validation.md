# A5 MTP Optimization, MX Adapter, And Non-A5 Boundary

This branch starts from upstream xllm-ops
`ed2172dd99fb676fcb5f4dbe0d97934019df0505`.
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
## MX Torch Adapter Ownership

`torch_adapter/` owns MX dynamic quantization, SwiGLU quantization, RMSNorm
quantization, Add+RMSNorm quantization, grouped matmul/SwiGLU quantization,
runtime capability checks, load-time fusion verification and the implementation
for builds without the complete MX API set. Public declarations are in
`torch_adapter/include/xllm_ops/mx_quantization.h`.

`cmake/mxfp8.cmake` detects Torch/Torch-NPU/CANN MX capabilities. xLLM includes
that module and attaches `torch_adapter/` to its existing `xllm_ops` host
library. The adapter uses the consumer's existing ACLNN invocation helper and
platform guard; this directory is not part of the standalone CANN OPP build.
Both enabled and unavailable implementations must be checked after a pointer
update. The model loader, Linear/MLP/GDN integration and architecture policy
remain in xLLM.

## Non-A5 Boundary

MXFP4 and MXFP8 are enabled only on Ascend950 (A5). A2/A3 keep their existing
FP8/BF16 paths, and the MX adapter is not selected on those devices. A
checkpoint that explicitly requires MX on a non-A5 device fails during loading
instead of silently taking an incompatible path.

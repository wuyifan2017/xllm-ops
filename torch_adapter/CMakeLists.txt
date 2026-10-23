# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Attach the adapters to xLLM's existing Torch/CANN host library.
# The consumer supplies its ACLNN invocation helper and platform guard.
if(NOT TARGET xllm_ops)
  message(FATAL_ERROR "MX Torch adapters require the xllm_ops host target")
endif()

if(XLLM_HAS_NPU_MXFP8)
  set(MXFP8_QUANT_SOURCE mxfp8_quant.cpp)
else()
  set(MXFP8_QUANT_SOURCE mxfp8_quant_unavailable.cpp)
endif()

target_sources(xllm_ops PRIVATE ${MXFP8_QUANT_SOURCE})
target_include_directories(xllm_ops PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

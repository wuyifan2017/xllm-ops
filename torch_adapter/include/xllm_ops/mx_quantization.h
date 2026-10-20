/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::npu {

bool mxfp8_runtime_available();

std::tuple<at::Tensor, at::Tensor> mxfp8_dynamic_quant(const at::Tensor& input);

std::tuple<at::Tensor, at::Tensor> mxfp8_swiglu_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit);

std::tuple<at::Tensor, at::Tensor> mxfp4_swiglu_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit);

std::tuple<at::Tensor, at::Tensor> mxfp8_rms_norm_quant(const at::Tensor& x,
                                                        const at::Tensor& gamma,
                                                        double epsilon);

std::tuple<at::Tensor, at::Tensor, at::Tensor> mxfp8_add_rms_norm_quant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon);

int64_t verify_mxfp8_swiglu_quant(const at::Tensor& x,
                                  const at::Tensor& ref_act,
                                  double clamp_limit);

std::tuple<at::Tensor, at::Tensor> mxfp8_grouped_matmul_swiglu_quant(
    const at::Tensor& x,
    const torch::TensorList& weight,
    const torch::TensorList& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list);

}  // namespace xllm::kernel::npu

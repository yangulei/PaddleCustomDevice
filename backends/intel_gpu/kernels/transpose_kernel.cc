// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include "dnn_support.hpp"
namespace custom_kernel {

template <typename T>
void TransposeKernel(const phi::Context& ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();

  auto x_data = x.data<T>();
  auto out_data = ctx.template Alloc<T>(out);
  show_kernel("TransposeKernel");
  show_debug("x{dims}=" << x.dims() << " x{rank}=" << x_dims.size()
                        << " out{dims}=" << out->dims());

  if (out->numel() == 0) {
    return;
  }
  auto rank = x_dims.size();
  if (rank == 1) {
    memcpy(out_data, x_data, x.numel() * sizeof(T));
  }
  PD_CHECK(axis.size() == rank,
           "axis.size (%d) must be equal the rank of input (%d).",
           axis.size(),
           rank);

  std::vector<size_t> step(out_dims.size(), 1);
  for (auto i = out_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * out_dims[i];
  }

  std::vector<size_t> index(rank, 0);
  for (auto i = 0; i < x.numel(); ++i) {
    std::vector<size_t> dst_index(rank, 0);
    for (auto j = 0; j < rank; ++j) {
      dst_index[j] = index[axis[j]];
    }
    out_data[phi::vec_product(dst_index, step)] = x_data[i];

    index.back()++;
    for (auto j = rank - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j - 1]++;
        index[j] = 0;
      } else {
        break;
      }
    }
  }
}

dnnl::memory::dims computeStrides(const std::vector<int64_t>& dims , const std::vector<int>& axis
) {
       size_t rank = axis.size();
       std::vector<int64_t> strides(rank);
       unsigned int total_stride = 1;
       for (int i = rank - 1; i >= 0; --i) {
         strides[axis[i]] = total_stride;
         total_stride *= dims[axis[i]];
       }
      show_debug("computeStrides strides=" << strides << " from [ dims="<< dims << " axis="<< axis << "]");
       return strides;
}

template <typename T>
void TransposeKernelGPU(const phi::Context& ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  show_kernel("TransposeKernelGPU ");

  using namespace dnnl;
  using tag = memory::format_tag;
  using dt = memory::data_type;
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(ctx.stream()));

  if (!q) {
  }

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  auto x_dims = x.dims();
  auto out_dims = out->dims();

  auto x_data = x.data<T>();
  auto out_data = ctx.template Alloc<T>(out);
  show_debug( "x{dims}="<< x.dims() <<  " x{rank}="<< x_dims.size() << " out{dims}="<< out->dims() << " axis="<< axis ) ;
  if (out->numel() == 0) {
    return;
  }
  auto rank = x_dims.size();
  if (rank == 1) {
  //  memcpy(out_data, x_data, x.numel() * sizeof(T));
  auto total_cpy_bytes = x.numel() * sizeof(T);
  q->submit([&](sycl::handler& h) {
    h.memcpy(out_data, x_data, total_cpy_bytes);
  });
  q->wait();
   return;
  }

  PD_CHECK(axis.size() == rank,
           "axis.size (%d) must be equal the rank of input (%d).",
           axis.size(),
           rank);
try {
  dnnl::memory::dims dims_src = x.dims();
  dnnl::memory::dims dims_dst = out->dims();


  // auto md_src = memory::desc(dims_src,
  //                            dnn_support::toDnnType<T>::type,
  //                            dnn_support::dims2Tag(dims_src));

  // auto md_dst = memory::desc(dims_src,
  //                            dnn_support::toDnnType<T>::type,
  //                            dnn_support::axis2Tag(axis));

  std::vector<int> logical_axis(dims_src.size(),0);
  for(auto i=0;i<logical_axis.size();++i)
  {
    logical_axis[i]=i;
  }
  show_debug("logical_axis=" << logical_axis << " axis=" << axis);
  auto md_src = memory::desc(dims_src,
                             dnn_support::toDnnType<T>::type,
                             computeStrides(dims_src, logical_axis));

  auto md_dst = memory::desc(
      dims_src, dnn_support::toDnnType<T>::type, computeStrides(dims_src,axis));

  auto mem_src = memory(md_src, eng, x_data);
  auto mem_dst = memory(md_dst, eng, out_data);

  auto reorder_pd =
      reorder::primitive_desc(eng, md_src, eng, md_dst);

  // Create the primitive.
  auto reorder_prim = reorder(reorder_pd);

  std::unordered_map<int, memory> reorder_args;
  reorder_args.insert({DNNL_ARG_SRC, mem_src});
  reorder_args.insert({DNNL_ARG_DST, mem_dst});

  // Primitive execution: reorder with scaled sum.
  reorder_prim.execute(engine_stream, reorder_args);
  engine_stream.wait();

} catch (std::exception& e) {
     show_debug(" Catch error=" << e.what());
     throw e;
}

show_debug("**** TRANSPOSE OK *** x=" << x.dims())


}





}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(transpose,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::TransposeKernel,
                    bool,
                    float,
                    double,
                    uint8_t,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t) {}
/*
PD_BUILD_PHI_KERNEL(transpose,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::TransposeKernelGPU,
                    bool,
                    float,
                    double,
                    uint8_t,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t) {}
*/

PD_BUILD_PHI_KERNEL(transpose,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::TransposeKernelGPU,
                    float  ) {}
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
#include <unistd.h>
#include "./dnn_support.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "paddle/phi/backends/device_ext.h"

#define MEMORY_FRACTION 0.5f

C_Status Init() {
  std::cout << "custom_cpu plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif
  return C_SUCCESS;
}

// **** Types *****
template <class T>
using up_t = std::unique_ptr<T>;

DeviceConfigPtr devconf;
std::mutex mx;
std::recursive_mutex rmux;

auto intel_match = [](const sycl::device &dev) -> bool {
  const auto name = dev.template get_info<sycl::info::device::name>();
  return (name.find("Intel(R) Graphics") != std::string::npos) ? true : false;
};

struct DeviceCtx {
  sycl::device _dev;
  std::vector<std::unique_ptr<sycl::queue>> _streams;
  bool _def_stream;
  size_t allocated_mem;
  size_t _dev_memory_size;
  DeviceCtx(sycl::device dev)
      : _dev{std::move(dev)},
        _def_stream{true},
        allocated_mem{0},
        _dev_memory_size(_dev.get_info<sycl::info::device::global_mem_size>()) {
  }

  sycl::queue *create_stream() {
    auto u_ptr = std::make_unique<sycl::queue>(_dev);
    _streams.push_back(std::move(u_ptr));

    return &(*(*(_streams.rbegin())));
  }

  sycl::queue *getDefaultOrCreate() {
    if (_def_stream && _streams.size()) {
      _def_stream = false;
      return _streams[0].get();
    }

    return create_stream();
  }

  sycl::queue &getStream(size_t index = 0) {
    if (!_streams.size()) create_stream();
    return *(_streams[index]);
  }

  void copy(sycl::queue &q, void *dst, const void *src, size_t size) {
    q.submit([&](sycl::handler &h) { h.memcpy(dst, src, size); });
    q.wait();
  }

  void copy(void *dst, const void *src, size_t size) {
    copy(getStream(), dst, src, size);
  }

  sycl::queue &getStream(C_Stream stream) {
    auto it = std::find_if(
        _streams.begin(), _streams.end(), [stream](auto &single_stream) {
          return single_stream.get() == reinterpret_cast<sycl::queue *>(stream);
        });

    if (it == _streams.end()) {
      show_error("*FATAL ERROR STREAM not found*");
    }
    return **it;
  }

  size_t getMemorySize() { return _dev_memory_size; }

  size_t getFreeMemorySize() { return (getMemorySize() - allocated_mem) / 8; }

  void alloc_mem(size_t _size) { allocated_mem += _size; }

  void free_mem(size_t _size) { allocated_mem -= _size; }
};

std::vector<DeviceCtx> reg_dev;

C_Status InitDevice(const C_Device device) {
  InitializeDevConf();
  show_debug("init-device : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  show_debug("set-device : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = 0;
  show_debug("get-device() : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  show_debug("destroy-device() : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  if (!reg_dev.size()) {
    InitializeDevConf();
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    std::copy_if(devices.begin(),
                 devices.end(),
                 std::back_inserter(reg_dev),
                 intel_match);

    if (!reg_dev.size()) {
      show_error("No Intel GPUs found");
      return C_FAILED;
    }
  }

  *count = reg_dev.size();
  show_debug("getdevicescount() count=" << *count);

  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  show_debug("getdeviceList() fill=" << reg_dev.size());

  for (size_t i = 0; i < reg_dev.size(); ++i) devices[i] = static_cast<int>(i);

  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  show_debug("async-memcpy  dst=" << dst << " src=" << src << " size=" << size);

  auto &dev_ctx = reg_dev[device->id];

  auto &dev_stream = dev_ctx.getStream(stream);

  dev_ctx.copy(dev_stream, dst, src, size);

  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  show_debug("memcpy-p2p  memcpy() dst=" << dst << " src=" << src
                                         << " size=" << size);
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  show_debug("async-memcpy-p2p  memcpy() dst=" << dst << " src=" << src
                                               << " size=" << size);
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  show_memory("request allocate size=" << size << " device=" << device->id);

  if (size > reg_dev[device->id].getFreeMemorySize()) {
    show_error("## No free memory INTERNAL ERROR OUT OF MEMORY requested size="
               << size << " left=" << reg_dev[device->id].getFreeMemorySize()
               << " ##");
    return C_FAILED;
  }

  auto &stream = reg_dev[device->id].getStream();

  *ptr = sycl::aligned_alloc_device(64, size, stream);
  // *ptr = sycl::aligned_alloc_shared(64, size, stream);

  if (!*ptr) {
    show_error("#### Error : Can't allocate memory size="
               << size << " free_mem_size="
               << reg_dev[device->id].getFreeMemorySize() << " ####");
    return C_FAILED;
  }

  reg_dev[device->id].alloc_mem(size);

  show_memory("allocate success size="
              << size << " left=" << reg_dev[device->id].getFreeMemorySize());

  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  show_memory("deallocate size=" << size);

  auto &stream = reg_dev[device->id].getStream();

  sycl::free(ptr, stream);

  reg_dev[device->id].free_mem(size);

  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  show_debug("create-stream for device=" << device->id);

  *stream = reinterpret_cast<C_Stream>(reg_dev[device->id].create_stream());

  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  show_debug("destroy-stream device->id=" << device->id
                                          << " stream=" << stream);

  auto &_streams = reg_dev[device->id]._streams;
  auto it = std::find_if(
      _streams.begin(), _streams.end(), [stream](auto &single_stream) {
        return single_stream.get() == reinterpret_cast<sycl::queue *>(stream);
      });

  if (it != _streams.end()) _streams.erase(it);

  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  show_debug("create-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  show_debug("record-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  show_debug("destroy-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  show_debug("sync-device devid=" << device->id);
  auto &dev_ctx = reg_dev[device->id];

  for (auto &stream : dev_ctx._streams) {
    stream->wait();  // ???????
  }
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  show_debug("sync-stream devid=" << device->id);
  auto ret_stream = reg_dev[device->id].getStream(stream);
  ret_stream.wait();

  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  show_debug("sync-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  show_debug("stream-wait-event devid=" << device->id);

  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) {
  show_debug("visible-devices devices=" << *devices);
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  auto &dev_ctx = reg_dev[device->id];
  *total_memory = dev_ctx.getMemorySize();
  *free_memory = dev_ctx.getFreeMemorySize();

  show_memory("device-mem-stat device=" << device->id
                                        << " TotalMemory=" << *total_memory
                                        << " FreeMemory=" << *free_memory);

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  show_memory("device-min-chunk-size device=" << device->id);
  InitializeDevConf();
  *size = devconf->chunk_size;
  return C_SUCCESS;
}

C_Status MemoryCopyH2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-h2d dst=" << dst << " src=" << src
                                     << " size=" << size);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

C_Status MemoryCopyD2H(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-d2h size=" << size << " dst=" << dst
                                      << " src=" << src);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

C_Status MemoryCopyD2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-d2d size=" << size << " dst=" << dst
                                      << " src=" << src);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

ccl::datatype ToOccl(const C_DataType data_type) {
  ccl::datatype dtype;
  if (data_type == C_DataType::UINT8) {
    dtype = ccl::datatype::uint8;
  } else if (data_type == C_DataType::UINT16) {
    dtype = ccl::datatype::uint16;
  } else if (data_type == C_DataType::UINT32) {
    dtype = ccl::datatype::uint32;
  } else if (data_type == C_DataType::UINT64) {
    dtype = ccl::datatype::uint64;
  } else if (data_type == C_DataType::INT8) {
    dtype = ccl::datatype::int8;
  } else if (data_type == C_DataType::INT16) {
    dtype = ccl::datatype::int16;
  } else if (data_type == C_DataType::INT32) {
    dtype = ccl::datatype::int32;
  } else if (data_type == C_DataType::INT64) {
    dtype = ccl::datatype::int64;
  } else if (data_type == C_DataType::FLOAT16) {
    dtype = ccl::datatype::float16;
  } else if (data_type == C_DataType::FLOAT32) {
    dtype = ccl::datatype::float32;
  } else if (data_type == C_DataType::FLOAT64) {
    dtype = ccl::datatype::float64;
  } else if (data_type == C_DataType::BFLOAT16) {
    dtype = ccl::datatype::bfloat16;
  } else {
    LOG(ERROR) << "Datatype " << data_type << " is not supported in oneCCL.";
  }
  return dtype;
}

ccl::reduction ToOccl(const C_CCLReduceOp op) {
  ccl::reduction reduction;
  if (op == C_CCLReduceOp::SUM) {
    reduction = ccl ::reduction::sum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    reduction = ccl ::reduction::prod;
  } else if (op == C_CCLReduceOp::MAX) {
    reduction = ccl ::reduction::max;
  } else if (op == C_CCLReduceOp::MIN) {
    reduction = ccl ::reduction::min;
  } else {
    LOG(ERROR) << "ReduceOp " << op << "is not supported in oneCCL.";
  }
  return reduction;
}

C_Status OcclGetUniqueIdSize(size_t *size) {
  *size = ccl::kvs::address_max_size;
  return C_SUCCESS;
}

C_Status OcclGetUniqueId(C_CCLRootId *unique_id) {
  auto kvs = ccl::create_main_kvs();
  auto kvs_addr = kvs->get_address();
  memcpy(unique_id->data, (void *)kvs_addr.data(), ccl::kvs::address_max_size);
  return C_SUCCESS;
}

C_Status OcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  ccl::kvs::address_type kvs_addr;
  memcpy((void *)kvs_addr.data(), unique_id->data, ccl::kvs::address_max_size);
  auto kvs = ccl::create_kvs(kvs_addr);
  auto ccl_comm = reinterpret_cast<ccl::communicator *>(comm);
  *ccl_comm = ccl::create_communicator(nranks, rank, kvs);
  return C_SUCCESS;
}

C_Status OcclDestroyComm(C_CCLComm comm) {
  if (comm) {
    delete reinterpret_cast<ccl::communicator *>(comm);
    comm = nullptr;
  }
  return C_SUCCESS;
}

C_Status OcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  auto ret_evt = ccl::allreduce(
      send_buf,
      recv_buf,
      count,
      ToOccl(data_type),
      ToOccl(op),
      *reinterpret_cast<ccl::communicator *>(comm),
      ccl::create_stream(*reinterpret_cast<sycl::queue *>(stream)));
  ret_evt.wait();
  return C_SUCCESS;
}

C_Status OcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  auto ret_evt = ccl::broadcast(
      buf,
      count,
      ToOccl(data_type),
      root,
      *reinterpret_cast<ccl::communicator *>(comm),
      ccl::create_stream(*reinterpret_cast<sycl::queue *>(stream)));
  ret_evt.wait();
  return C_SUCCESS;
}

C_Status OcclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream) {
  auto ret_evt =
      ccl::reduce(send_buf,
                  recv_buf,
                  count,
                  ToOccl(data_type),
                  ToOccl(op),
                  root,
                  *reinterpret_cast<ccl::communicator *>(comm),
                  ccl::create_stream(*reinterpret_cast<sycl::queue *>(stream)));
  ret_evt.wait();
  return C_SUCCESS;
}

C_Status OcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  auto occl_comm = reinterpret_cast<ccl::communicator *>(comm);
  std::vector<size_t> recv_counts(occl_comm->size(), count);
  auto ret_evt = ccl::allgatherv(
      send_buf,
      count,
      recv_buf,
      recv_counts,
      ToOccl(data_type),
      *occl_comm,
      ccl::create_stream(*reinterpret_cast<sycl::queue *>(stream)));
  ret_evt.wait();
  return C_SUCCESS;
}

C_Status OcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  auto ret_evt = ccl::reduce_scatter(
      send_buf,
      recv_buf,
      count,
      ToOccl(data_type),
      ToOccl(op),
      *reinterpret_cast<ccl::communicator *>(comm),
      ccl::create_stream(*reinterpret_cast<sycl::queue *>(stream)));
  ret_evt.wait();
  return C_SUCCESS;
}

C_Status OcclGroupStart() {
  LOG(ERROR) << "xccl_group_start is not supported in oneCCL.";
  return C_ERROR;
}

C_Status OcclGroupEnd() {
  LOG(ERROR) << "xccl_group_end is not supported in oneCCL.";
  return C_ERROR;
}

C_Status OcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  LOG(ERROR) << "xccl_send is not supported in oneCCL.";
  return C_ERROR;
}

C_Status OcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  LOG(ERROR) << "xccl_recv is not supported in oneCCL.";
  return C_ERROR;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "intel_gpu";
  params->sub_device_type = "v0.1";
  show_debug("init-plugin " << params->device_type);

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DestroyDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  // params->interface->memory_copy_h2d = MemCpy;
  params->interface->memory_copy_h2d = MemoryCopyH2D;
  // params->interface->memory_copy_d2d = MemCpy;
  params->interface->memory_copy_d2d = MemoryCopyD2D;

  params->interface->memory_copy_d2h = MemoryCopyD2H;

  params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2h = AsyncMemCpy;
  params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = Allocate;
  params->interface->unified_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = Deallocate;
  params->interface->unified_memory_deallocate = Deallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->xccl_get_unique_id_size = OcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = OcclGetUniqueId;
  params->interface->xccl_comm_init_rank = OcclCommInitRank;
  params->interface->xccl_destroy_comm = OcclDestroyComm;
  params->interface->xccl_all_reduce = OcclAllReduce;
  params->interface->xccl_broadcast = OcclBroadcast;
  params->interface->xccl_reduce = OcclReduce;
  params->interface->xccl_all_gather = OcclAllGather;
  params->interface->xccl_reduce_scatter = OcclReduceScatter;
  params->interface->xccl_group_start = OcclGroupStart;
  params->interface->xccl_group_end = OcclGroupEnd;
  params->interface->xccl_send = OcclSend;
  params->interface->xccl_recv = OcclRecv;
}

# Intel Arc Battlemage (Xe2) on Linux: the inference layers, and when to use each

This page is a survey, not a port. Strata does not run on Intel graphics cards yet. It describes how inference runs on
Battlemage cards under Linux today, layer by layer, how fast each layer was measured to be and by whom, and what a
Strata backend for these cards would be built on. **None of the numbers below were measured by the Strata
maintainers.** Each one names its source, card and software. They were collected in October 2026, and Intel's
stack changes month to month, so check the dates.

## In short

- **To run models today:** for GGUF models, use **llama.cpp's SYCL backend** (oneAPI 2026.0, the `xe` kernel driver,
  `GGML_SYCL_F16=ON`). It produced the most tokens per second in the 2026 community table. Use **Vulkan** (Mesa 26.1
  or newer) when you want no oneAPI install, or when prompt speed matters more. For many users and batched serving,
  use **vLLM on XPU**, which only helps with quant formats that have a native XPU kernel (MXFP4, GPTQ/AWQ int4, FP8).
  **OpenVINO GenAI** is the best fit for Intel-converted int4 models and is fast on small dense models.
- **Avoid:** IPEX (Intel Extension for PyTorch, archived 30 March 2026) and ipex-llm (archived 28 January 2026).
  Use upstream PyTorch XPU and upstream vLLM instead.
- **To write a fast engine (Strata):** use **SYCL (DPC++)** as the main language and **Level Zero** for memory and
  queue control. Use **ESIMD or inline vISA** only for the few kernels where XMX (the matrix engines) or exact
  register control pays off. **oneMKL / oneDNN** cover the dense prefill GEMMs. [For Strata](#for-strata) maps every
  CUDA feature the engine uses to its Intel counterpart.
- **On Linux, the setting that matters most:** use the `xe` kernel driver, not `i915`. The llama.cpp SYCL maintainer
  wrote that "Xe driver gives a huge boost over i915" (llama.cpp
  [discussion #23313](https://github.com/ggml-org/llama.cpp/discussions/23313)). Also enable Resizable BAR and run a
  current compute-runtime / Mesa.

## The cards

| Card | Xe cores (XMX engines) | VRAM, bus | Bandwidth | PCIe | Board power | INT8 TOPS (dense) |
|---|---|---|---|---|---|---|
| Arc B570 | 18 (144) | 10 GB GDDR6, 160-bit | 380 GB/s | 4.0 x8 | 150 W | 203 |
| Arc B580 | 20 (160) | 12 GB GDDR6, 192-bit | 456 GB/s | 4.0 x8 | 190 W | 233 |
| Arc Pro B50 | 16 (128) | 16 GB GDDR6, 128-bit | 224 GB/s | 5.0 x8 | 70 W | 170 |
| Arc Pro B60 | 20 (160) | 24 GB GDDR6, 192-bit | 456 GB/s | 5.0 x8 | 120-200 W | 197 |
| Arc Pro B65 | 20 (160) | 32 GB GDDR6 ECC, 256-bit | 608 GB/s | 5.0 x16 | 200 W | 197 |
| Arc Pro B70 | 32 (256) | 32 GB GDDR6, 256-bit | 608 GB/s | 5.0 x16 | 230 W (partners 160-290 W) | 367 |

Sources: Intel's [B60](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b60-gpu.pdf)
and [B65](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b65-gpu.pdf)
datasheets, the [B50 spec page](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html),
[Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced) and
[StorageReview](https://www.storagereview.com/review/intel-arc-pro-b70-review) for the B70 ($949 list price, over $1,100
in shops at review time), [iFixit](https://www.ifixit.com/Device/Intel_Arc_B570) for the B570. The B580's TOPS and
board power are Intel's launch figures. The B570/B580/B60 use the BMG-G21 chip; the B65/B70 use the larger BMG-G31. No consumer B770 has been
announced.

For a MoE engine that keeps all experts in RAM and caches the busiest ones in VRAM, three of these numbers matter:

- **VRAM decides how many experts sit on the card.** On NVIDIA, Strata measured ~700 more experts per extra GB
  ([how it works](HOW_IT_WORKS.md)). By that measure, a 24 GB B60 or 32 GB B65/B70 holds far more of the model than a
  12 GB B580.
- **The PCIe link decides how fast prompts stream experts.** The B570/B580 have only **PCIe 4.0 x8**, about 16 GB/s
  per direction in theory and less in practice; no measured host-to-device figure for Battlemage was found. That is
  half the link of a PCIe 4.0 x16 card. The B65/B70 have PCIe 5.0 x16 (about 64 GB/s theoretical).
- **Memory bandwidth bounds decode.** The B580 and B60 have 456 GB/s, about half an RTX 4090 and close to an RX 7800 XT.

The architecture, briefly (Intel's and [TechPowerUp's](https://www.techpowerup.com/review/intel-arc-b580-battlemage-unboxing-preview/2.html)
descriptions):

- Each Xe core has 8 vector engines (XVE) and 8 XMX matrix engines of 2048 bits each.
- The vector engines are natively **SIMD16**. Alchemist's were SIMD8.
- XMX supports INT2 to INT8, FP16, BF16 and TF32.
- Kernels run with sub-group (warp) size **16 or 32**. Size 8 is gone on Xe2.
- XMX is reached through the DPAS instruction.

## The layers, bottom to top

```
 apps / servers   llama.cpp (SYCL | Vulkan)   vLLM-XPU   OpenVINO GenAI / OVMS   PyTorch XPU   (Strata?)
 kernel libs      oneDNN   oneMKL   sycl-tla (CUTLASS for SYCL)   vllm-xpu-kernels   Triton XPU
 languages        SYCL / DPC++  (+ ESIMD, joint_matrix, inline vISA)    OpenCL C    GLSL/Slang->SPIR-V   HIP via chipStar
 runtime APIs     Unified Runtime -> Level Zero        OpenCL              Vulkan (Mesa ANV)
 user driver      compute-runtime "NEO" + IGC (graphics compiler)          Mesa ANV
 kernel driver    xe  (i915 still binds older cards; not recommended)  + GuC/HuC firmware
```

### 1. Kernel driver: `xe`

- **What it is:** Intel's current DRM driver, the default for Battlemage in mainline kernels. On a recent Fedora or
  Ubuntu the B580 binds to `xe` without any configuration
  ([Fedora discussion](https://discussion.fedoraproject.org/t/intel-arc-b580-on-fedora/183238)).
- **Use:** always. If both `i915` and `xe` are loaded, keep only `xe`. The llama.cpp SYCL maintainer calls `i915`
  "not recommended due to its old driver" ([#23313](https://github.com/ggml-org/llama.cpp/discussions/23313)).
- **Check:**
  - `lspci -k` shows `Kernel driver in use: xe`.
  - **Resizable BAR is on.** Intel says it is
    [required for optimal performance](https://www.intel.com/content/www/us/en/support/articles/000091128.html).
    `dmesg | grep -i bar` shows whether the driver could resize BAR2. On some boards it fails with `-ENOSPC`; see
    [Jeff Geerling's notes](https://www.jeffgeerling.com/blog/2025/resizeable-bar-support-on-raspberry-pi/).
  - For llama.cpp's free-memory query, set `ZES_ENABLE_SYSMAN=1`.

### 2. User-mode driver: compute-runtime (NEO) and IGC

- **What it is:** [intel/compute-runtime](https://github.com/intel/compute-runtime) provides both Level Zero and
  OpenCL. IGC compiles SPIR-V and vISA to the GPU's ISA.
- **Use:** with every compute path except Vulkan.
- **Minimum versions:** Battlemage needs **Level Zero ≥ 1.6.33578 and compute-runtime ≥ 25.18.33578**. Older images
  see only the integrated GPU ([ipex-llm #13324](https://github.com/intel/ipex-llm/issues/13324)). Reporters in
  2026 used Level Zero 1.14-1.15.
- **Watch for:** BMG-specific bugs still being fixed. One example is "could not create an engine" on a B580 matmul
  ([compute-runtime #949](https://github.com/intel/compute-runtime/issues/949)). Phoronix also reported higher kernel-launch latency than
  Alchemist in 2025, which matters for decode, where each token launches hundreds of small kernels.

### 3. Level Zero

- **What it is:** Intel's low-level GPU API, the CUDA-driver-API equivalent. It provides:
  - devices, contexts and USM allocations (`zeMemAllocDevice/Host/Shared`)
  - command queues and command lists, both regular (recorded once, replayed) and immediate
  - events, fences, IPC and peer-to-peer access
  - the separate copy engines (BCS) for transfers that overlap compute
  - Sysman for telemetry
- **Use directly** where SYCL hides a control you need:
  - choosing the copy engine for H2D streaming
  - importing existing host memory (an `mmap`ed model file) for GPU access
  - reading power and temperature for a monitor
  - building and replaying a command list yourself instead of through SYCL Graph

  Every SYCL object can hand back its Level Zero handle (`sycl::get_native<backend::ext_oneapi_level_zero>`; see the
  [backend spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)),
  so mixing the two is normal.
- **Don't write kernels in it.** Level Zero runs SPIR-V or native binaries that some other compiler produced.

### 4. OpenCL

- **What it is:** the same NEO driver behind the OpenCL 3.0 API, with Intel's subgroup and DPAS extensions
  (`cl_intel_subgroup_matrix_multiply_accumulate` and others).
- **Use:** for portable code that already exists in OpenCL C, or for tools like `clpeak`.
- **New engines should prefer SYCL plus Level Zero.** OpenCL has no USM-first memory model, its command-graph
  extension is less mature, and Intel's own libraries (oneDNN, oneMKL) and llama.cpp target Level Zero.
- **Pitfall:** PyTorch's probe of the OpenCL extension flags wrongly reports no XMX on a B580
  ([pytorch #196074](https://github.com/pytorch/pytorch/issues/196074)). Don't decide XMX support from those flags.

### 5. Vulkan (Mesa ANV)

- **What it is:** Mesa's open Intel Vulkan driver, with compute shaders compiled from GLSL, HLSL or Slang to SPIR-V.
  ANV has exposed `VK_KHR_cooperative_matrix` (XMX) since Mesa 24.0
  ([Phoronix](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix)). `VK_EXT_external_memory_host` lets the
  GPU use existing host memory.
- **Measured:**
  - **Mesa version matters more than the kernel version.** Phoronix's Linux 7.1 test on a B580 found that the new
    kernel barely changed llama.cpp-Vulkan speed, while Mesa 26.1 raised text generation noticeably (reported
    second-hand by [Jonathan Mann's B70 tests](https://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/)).
  - A community image cites a B580 going from about 30 to 67 tok/s with Mesa 26.1's cooperative-matrix path
    ([snailium/llama.cpp-vulkan](https://github.com/snailium/llama.cpp-vulkan); not verified here).
  - **llama-2-7b Q4_0, llama.cpp, mid-2026**
    ([#23313](https://github.com/ggml-org/llama.cpp/discussions/23313) comments). Vulkan and SYCL trade wins by card:

    | Card | Vulkan pp512 / tg128 (t/s) | SYCL pp512 / tg128 (t/s) | Faster |
    |---|---|---|---|
    | B580 | ~1930-2015 / 73-77 | ~1930-2015 / 82-87 | SYCL decode |
    | B50 | ~590 / 40-42 | ~397 / 46-48 | Vulkan prompts, SYCL decode |

  - In early 2025 Vulkan was about twice as fast as SYCL on the B580
    ([ipex-llm #12991](https://github.com/intel/ipex-llm/issues/12991)). That no longer holds.
- **Use:**
  - when you want the same code on NVIDIA, AMD and Intel with no vendor SDK
  - when the install must stay small (a distro Mesa, no oneAPI)
  - for prompt-heavy work on the smaller cards
- **Limits:**
  - Kernels are written in a shader language, with no C++ templates.
  - You don't control sub-group size, register allocation or the copy engine the way you do in SYCL or Level Zero.
  - Cooperative matrix has caused GPU timeouts on some Intel drivers. Those reports were on Windows and a laptop
    chip ([llama.cpp #20554](https://github.com/ggml-org/llama.cpp/issues/20554)).

### 6. SYCL / DPC++ (oneAPI)

- **What it is:** C++ single-source kernels (`icpx -fsycl`), with Intel's open-source
  [intel/llvm](https://github.com/intel/llvm) or the oneAPI 2026 release. The Unified Runtime turns SYCL calls into
  Level Zero calls. **On Xe2 the default is the Level Zero v2 adapter, which uses immediate command lists only**
  ([DPC++ environment variables](https://intel.github.io/llvm/EnvironmentVariables.html)).
- **Main tools:**
  - USM (`malloc_device`, `malloc_host`, `malloc_shared`)
  - in-order queues (closest to a CUDA stream), events
  - `reqd_sub_group_size(16|32)` and the `sycl::reduce_over_group` / `shift_group_left` / `permute_group_by_xor`
    shuffles
  - local memory (SLM)
  - **SYCL Graph** (`sycl_ext_oneapi_graph`, the CUDA Graphs equivalent, recorded into Level Zero command lists)
  - `prepare_for_device_copy` (`sycl_ext_oneapi_copy_optimize`), which imports existing host memory for fast copies
    ([Intel GPU optimization guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html))
- **Use:** as the default language for new compute code on Intel. It is what llama.cpp-SYCL, vLLM's XPU kernels,
  oneDNN, oneMKL and sycl-tla are written in.
- **Build flags:**
  - Compile ahead of time for the card: `-fsycl-targets=intel_gpu_bmg_g21` (B570/B580/B60) or
    `intel_gpu_bmg_g31` (B65/B70). Without them, the first launch compiles from SPIR-V; `SYCL_CACHE_PERSISTENT=1`
    caches that JIT result.
- **Measured, llama.cpp SYCL, llama-2-7b Q4_0, pp512 / tg128 t/s, FP16 build, flash attention off**
  ([#23313](https://github.com/ggml-org/llama.cpp/discussions/23313), community, not double-checked):

  | Card | pp512 | tg128 | Reported |
  |---|---|---|---|
  | B70 | 2830 | 106 | 2026-08 |
  | B580 | 1954-2064 | 74-84 | 2026-05/06 |
  | B60 | 1869 | 80 | 2026-08 |
  | B570 | 1356 | 73 | 2026-06 |
  | B50 | ~400 | ~47 | 2026-05 |

  The B70 also ran Qwen3.6-27B Q5_K_M at 936 / 22.9.

  The same discussion reports these patterns:
  - The FP16 build reads prompts 1.5-3x faster than the FP32 build, with about the same decode.
  - Turning flash attention on in that build halves prompt speed and adds a few percent to decode.
  - A second B580 or B70 split across cards did not help a 7B model.
- **Pitfalls on Battlemage (all 2026):**
  - wrong output from llama.cpp's Xe2 "reorder" optimization on a B70 unless `GGML_SYCL_DISABLE_OPT=1`
    ([#21893](https://github.com/ggml-org/llama.cpp/issues/21893), closed with no fix linked)
  - a crash on the first submit through the graph-capture path on a B70
    ([intel/llvm #22958](https://github.com/intel/llvm/issues/22958))
  - a segfault on a second USM memset in a default out-of-order queue with the v2 adapter
    ([intel/llvm #23034](https://github.com/intel/llvm/issues/23034))

  Use in-order queues, pin the oneAPI version, and test each upgrade.

#### 6a. ESIMD (explicit SIMD)

- **What it is:** a SYCL extension (`sycl::ext::intel::esimd`) where one work-item is a whole hardware thread
  operating on explicit vectors. It gives:
  - block loads and stores (2D block I/O)
  - direct DPAS (`xmx::dpas`)
  - named barriers
  - control over register use (large-GRF mode)
- **Use:** for the few kernels where you need the last 20-40%: quantized GEMV/GEMM with on-the-fly dequantization,
  attention, fused MoE. One 2026 project runs EXL3 trellis-quant GEMMs on Battlemage as ESIMD kernels inside vLLM
  ([exl3xpu](https://github.com/sybil-solutions/exl3xpu)).
- **Don't use it** for general code. It is Intel-only, verbose, and its CUDA-style per-thread code does not carry over.

#### 6b. `joint_matrix` (SYCL matrix extension)

- **What it is:** the portable way to reach XMX from SYCL, like CUDA's WMMA
  ([Intel guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/programming-intel-xmx-using-sycl-joint-matrix.html)).
- **On Xe2, in practice** ([candle SYCL PR](https://github.com/tobert/candle/pull/1)):
  - only FP16/BF16 at 16×16×16 with sub-group size 16
  - no INT8 or FP8 through `joint_matrix` there
  - an IGC 2026.1 crash on one permute
- **Use:** for FP16/BF16 tiles when you want portable SYCL. For INT8 DPAS, use ESIMD's `xmx::dpas` or inline vISA. The
  [TernSYCL](https://github.com/Torchit1/TernSYCL) kernels take that inline-vISA route for an int2×int8 DPAS with K=32.

### 7. Kernel libraries

| Library | What it gives on Battlemage | Use it for |
|---|---|---|
| **oneMKL** (`oneapi::mkl::blas::gemm`, batched GEMM) | cuBLAS-style dense GEMM in FP32, FP16, BF16 and INT8 | the drop-in for `cublasGemmEx` in prefill |
| **oneDNN** (matmul primitive) | weight-only quantized matmul: int8 activations with int4/int8 weights and grouped scales (3.7), FP4 weights and better int4-weight int8 matmul ([2026 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html)) | quantized prefill GEMMs without writing your own. A stream cached on the first queue broke graph replay in one integration ([auto-round #2206](https://github.com/intel/auto-round/issues/2206)) |
| **sycl-tla** (CUTLASS for SYCL) | CuTe-style GEMM, grouped GEMM (also FP16 × U4 mixed), flash attention. v0.8 added BMG-G31, with flash-attention BF16 at ~78% of peak on BMG; 0.9.3 fixed a BMG GEMM regression ([releases](https://github.com/intel/sycl-tla/releases)) | custom fused GEMMs and grouped MoE GEMM in prefill. The closest thing to writing CUTLASS on NVIDIA |
| **vllm-xpu-kernels** | fused MoE (bf16, fp8, mxfp8, mxfp4, int4), Xe2-tuned grouped GEMM, paged attention, MLA decode ([releases](https://github.com/vllm-project/vllm-xpu-kernels/releases)) | reading or borrowing MoE kernels; replaced IPEX inside vLLM |
| **Triton XPU** ([intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)) | Triton kernels compiled for Xe with DPAS and 2D block loads | Python-side custom kernels in PyTorch. Not for a C++ engine |

### 8. Frameworks and servers

| Layer | Status (Oct 2026) | When to use it |
|---|---|---|
| **llama.cpp, SYCL backend** | Active. Build with `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg_g21`, run with `ONEAPI_DEVICE_SELECTOR=level_zero:0` | GGUF models, one user, best decode on most cards (table above). MoE models with experts on the CPU (`--cpu-moe` / `-ot`) work through the generic offload |
| **llama.cpp, Vulkan backend** | Active. Needs Mesa 26.1+ for the fast cooperative-matrix path | No oneAPI wanted; prompt-heavy work on the B50; a mixed-vendor PC |
| **vLLM (XPU)** | Upstream. MoE on XPU since 0.10.2. vLLM claims 80%+ MoE hardware efficiency on the B-series ([blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b), vendor claim). The [ai-containers](https://github.com/intel/ai-containers/blob/main/vllm/0.14.1-xpu.md) release for PyTorch 2.10 was the last with IPEX | Many users, batching, tensor parallel over 2-8 B60/B70. **Only for formats with an XPU kernel:** MXFP4, GPTQ/AWQ int4, FP8. Most quantized MoE GGUFs fall to an NVIDIA-only path ([4× B60 report](https://ninabot.ch/en/anchor/benchmarks/)). GPTQ-int4 was the fast path on a B70 ([cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md)) |
| **OpenVINO GenAI / OVMS** | Active, Intel's own. INT4 weight compression through NNCF | Small/medium dense models converted to OpenVINO IR. Fast on a B580: Qwen3.5 9B int4 ~4000 t/s prefill, ~60 t/s decode before a nightly regression to ~1450 / 52 ([openvino #36061](https://github.com/openvinotoolkit/openvino/issues/36061)). No GGUF i-quants, no CPU-resident MoE experts |
| **PyTorch XPU** (upstream, `torch.xpu`) | Upstream since 2.5, the replacement for IPEX | Research, fine-tuning, Hugging Face Transformers. Not the fastest way to serve |
| **IPEX, ipex-llm** | **Archived** (IPEX on 2026-03-30, ipex-llm on 2026-01-28; [IPEX notice](https://pytorch-extension.intel.com/)). ipex-llm "has known security issues" | Not for new work |

### 9. Translation layers (CUDA or HIP code on Intel)

| Tool | What it does | When it fits |
|---|---|---|
| **SYCLomatic** (`c2s`, the "DPC++ Compatibility Tool") | Rewrites CUDA source to SYCL. Intel cites 90-95% automatic migration; the rest is by hand ([intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html), [GitHub](https://github.com/oneapi-src/SYCLomatic)) | The first pass of a real port. It produces readable SYCL that you then own and tune. Inline PTX (`mma.sync`, `cp.async`, `ldmatrix`) is not migrated |
| **chipStar** | Compiles unmodified HIP/CUDA to SPIR-V and runs it on OpenCL or Level Zero. v1.3 has a bundled HIP 7.2 stack ([releases](https://github.com/CHIP-SPV/chipStar/releases)) | A one-day check that an existing HIP code base builds and gives correct answers on Intel. Expect to lose speed where the code relies on wave32 shuffles, `__byte_perm`, dp4a or vendor BLAS. No measured LLM numbers on Battlemage were found |

## Which layer, when

| Goal | Use | Why |
|---|---|---|
| Run a GGUF model for one person, today | llama.cpp SYCL (or Vulkan) | Best measured decode; Vulkan when you want no oneAPI install |
| Serve many users, several cards | vLLM XPU with an int4/MXFP4/FP8 checkpoint | Batching, tensor parallel, native XPU MoE kernels |
| Smallest install, mixed-vendor PC | llama.cpp Vulkan on Mesa 26.1+ | Distro packages only |
| Intel-optimized small dense model | OpenVINO GenAI | Intel's int4 path; highest measured prompt speed on a B580 |
| Write a new engine or port a CUDA one | SYCL + Level Zero interop | Full control of queues, memory and sub-groups; Intel's libraries plug in |
| Squeeze a hot quantized kernel | ESIMD or inline vISA (DPAS, block I/O) | Explicit registers and INT8 XMX, which `joint_matrix` lacks on Xe2 |
| Dense GEMM in prefill | oneMKL (simple) or sycl-tla / oneDNN (fused or quantized) | Like cuBLAS / CUTLASS |
| Check that HIP code runs at all | chipStar | No source changes |

## Getting the most out of a Battlemage card on Linux

These are common to every layer. The numbers are from the sources above.

1. **Use the `xe` driver with Resizable BAR on** (see [layer 1](#1-kernel-driver-xe)).
2. **Run a current compute-runtime, IGC and Level Zero** (≥ 25.18 / 1.6.33578 just to see the card). For Vulkan, use
   Mesa ≥ 26.1. Distro packages lag; Intel's PPA or Fedora's latest are newer.
3. **Compile for the card ahead of time** (`intel_gpu_bmg_g21` / `bmg_g31`) and keep `SYCL_CACHE_PERSISTENT=1`, so
   the first request doesn't pay for JIT.
4. **Pick the card explicitly** with `ONEAPI_DEVICE_SELECTOR=level_zero:N`. An integrated Intel GPU otherwise shows up
   too, and some containers saw only that one ([ipex-llm #13324](https://github.com/intel/ipex-llm/issues/13324)).
5. **Prefer FP16 compute for prompts.** In llama.cpp it read prompts 1.5-3x faster than FP32 with the same decode
   ([#23313](https://github.com/ggml-org/llama.cpp/discussions/23313)).
6. **Cut launch overhead in decode** with in-order queues, one queue per stream of work, and a recorded graph
   (SYCL Graph or a Level Zero regular command list) for the per-token kernel chain. Battlemage's launch latency has
   been reported higher than NVIDIA's and AMD's.
7. **Keep data on the card.** The B570/B580's PCIe 4.0 x8 is the narrowest link of any card Strata supports today.
8. **Pin your stack and re-measure on every upgrade.** Several 2026 reports are regressions between builds, both in
   llama.cpp and in OpenVINO nightlies.

## For Strata

Strata's engine is CUDA (`src/kernels/cuda/*.cu`, 44 files) compiled for AMD through a thin HIP shim
(`include/strata/hip_compat`). Below are the CUDA features it uses, counted in `src/` and `include/`, and their Intel
counterparts:

| CUDA in Strata (uses) | Intel counterpart | Notes |
|---|---|---|
| `cudaMalloc/Memcpy/Memset`, streams, events (~2,000) | USM `malloc_device`, in-order `sycl::queue`, `sycl::event`; or Level Zero directly | Mechanical; SYCLomatic does most of it |
| CUDA Graphs (`cudaStreamBeginCapture`, `cudaGraphLaunch`, ~200) | `sycl_ext_oneapi_graph` (record and replay, or queue capture), or Level Zero regular command lists | Watch the 2026 capture crash on a B70 ([intel/llvm #22958](https://github.com/intel/llvm/issues/22958)). The decode loop depends on this |
| `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` (KV and expert data the GPU reads from RAM) | `sycl::malloc_host` / `zeMemAllocHost`: the device pointer is the host pointer | Kernels read host USM over PCIe, like mapped memory on CUDA |
| `cudaHostRegister` over `mmap`ed experts (13) | `prepare_for_device_copy` (`sycl_ext_oneapi_copy_optimize`), or Level Zero's external host-pointer import | Needed for `--mmap-experts`. The Level Zero import's exact availability on BMG is not confirmed here: test it first |
| `cublasGemmEx`, `cublasLt` (dense prefill) | oneMKL `gemm` (FP16/BF16), oneDNN matmul | AMD needed per-architecture hipBLASLt tables; check oneMKL's speed before tuning |
| Warp-32 shuffles and ballots (`__shfl_*_sync` ~117, `__ballot_sync`) | sub-group functions with `[[sycl::reqd_sub_group_size(32)]]` | Xe2 runs sub-group 32 natively, so the 32-lane logic carries over. Sub-group 16 may be faster but changes every reduction |
| `__dp4a` (int8 dot in the i-quant/Q8 kernels) | the DP4A instruction Xe2 has, reached through SYCL integer intrinsics or ESIMD | Check the generated ISA (`IGC_ShaderDumpEnable`): a scalar fallback would cost several times the instructions, as on HIP before `sudot4` |
| `__byte_perm` (35), `__vsub4` (20), `__ldg` (51) | SWAR versions already in `hip_compat/intrinsics.hpp`; `__ldg` becomes a plain load | The HIP shim's portable code can be reused as is |
| `mma.sync`, `ldmatrix`, `cp.async` (inline PTX, QSA path) | `joint_matrix` (FP16/BF16) or ESIMD `xmx::dpas` + 2D block loads | HIP already runs this through an ordered FP32 fallback; an Intel port can start the same way |
| Dynamic shared memory opt-in (`cudaFuncAttributeMaxDynamicSharedMemorySize`) | `sycl::local_accessor`; check the limit with `info::device::local_mem_size` | Kernels sized for more SLM than Xe2 gives need a smaller tile |
| NVML / sysfs telemetry for the Monitor | Level Zero Sysman (`ZES_ENABLE_SYSMAN=1`) or `xe` hwmon in sysfs | Monitor only |

**How a port could go**, cheapest first:

1. **Feasibility in a day:** build the HIP backend with **chipStar** against Level Zero and run the existing parity
   tests (`src/kernels/*_parity.cpp`) on a B580 or B60.
   - **Shows:** which kernels are correct with no changes, and whether host-mapped expert memory works.
   - **Doesn't show:** speed.
2. **The real backend:** run **SYCLomatic** over `src/kernels/cuda` and the runtime calls, then own the SYCL it
   produces.
   - **Keep** sub-group size 32 at first, so the warp logic and the parity tests carry over.
   - **Host memory:** Level Zero interop for host-memory import and the copy engine.
   - **Prefill:** oneMKL for the dense GEMMs.
   - **Decode loop:** SYCL Graph for the per-token kernel chain.
   - This mirrors how the HIP backend shares one source tree, but SYCL does not compile CUDA syntax, so it is a
     source fork of the kernels or a macro layer much larger than `hip_compat`.
3. **Speed:** measure first. Then move the hottest decode kernels to sub-group 16 or ESIMD:
   - the expert GEMV (`s2_gemv*`, `native_mmvq`)
   - the router
   - QSA attention

   Use XMX (DPAS) for the prefill MMQ and the grouped expert GEMM, borrowing from sycl-tla's grouped GEMM and
   vllm-xpu-kernels' fused MoE.
4. **Vulkan instead** would avoid oneAPI in the installer, but it means rewriting all 44 kernel files in a shader
   language. It only makes sense if Strata wanted one more backend for every vendor.

**What to expect, as an estimate rather than a measurement:**

- **Decode on the B580 and B60** should sit near the RX 7800 XT class, since the memory bandwidth is similar
  (456 GB/s), if the decode kernels reach a similar share of it. The llama.cpp SYCL numbers above suggest they can:
  the B580's 74-84 tok/s on a 7B Q4_0 model.
- **Card size:** the B60 (24 GB), B65 and B70 (32 GB) hold enough experts to matter.
- **On the B580:** 12 GB and a PCIe 4.0 x8 link put it at the low end for Strata's prompt path, which streams the next
  layer's experts over PCIe.

Any real number needs a measured run; this page will be updated with one when a port exists.

## Sources

Hardware:
- Intel datasheets: [Arc Pro B60](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b60-gpu.pdf),
  [Arc Pro B65](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b65-gpu.pdf),
  [Arc Pro B50](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html)
- B70: [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced),
  [StorageReview](https://www.storagereview.com/review/intel-arc-pro-b70-review)
- Architecture: [TechPowerUp B580 architecture](https://www.techpowerup.com/review/intel-arc-b580-battlemage-unboxing-preview/2.html),
  [Tom's Hardware B580 review](https://tomshardware.com/pc-components/gpus/intel-arc-b580-review-the-new-usd249-gpu-champion-has-arrived)

llama.cpp:
- [SYCL performance discussion #23313](https://github.com/ggml-org/llama.cpp/discussions/23313)
- [Arc status #12570](https://github.com/ggml-org/llama.cpp/discussions/12570)
- Issues: [#21893](https://github.com/ggml-org/llama.cpp/issues/21893), [#20554](https://github.com/ggml-org/llama.cpp/issues/20554)
- [Vulkan coopmat PR #10597](https://github.com/ggml-org/llama.cpp/pull/10597)

Mesa / Vulkan:
- [Jonathan Mann, B70 llama.cpp tests](https://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/)
- [Phoronix, ANV cooperative matrix](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix)

oneAPI / SYCL:
- [intel/llvm environment variables](https://intel.github.io/llvm/EnvironmentVariables.html)
- [Level Zero backend extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)
- [Intel GPU optimization guide: data transfers](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html)
- [joint_matrix guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/programming-intel-xmx-using-sycl-joint-matrix.html)
- intel/llvm issues: [#22958](https://github.com/intel/llvm/issues/22958), [#23034](https://github.com/intel/llvm/issues/23034)
- [compute-runtime #949](https://github.com/intel/compute-runtime/issues/949)

Kernel libraries:
- [sycl-tla](https://github.com/intel/sycl-tla)
- [oneDNN 2026 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html)
- [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels/releases)
- [Triton XPU](https://github.com/intel/intel-xpu-backend-for-triton)
- Field reports: [exl3xpu](https://github.com/sybil-solutions/exl3xpu), [TernSYCL](https://github.com/Torchit1/TernSYCL),
  [candle SYCL PR](https://github.com/tobert/candle/pull/1)

vLLM:
- [vLLM on Arc Pro B-series](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)
- [4× B60 benchmarks](https://ninabot.ch/en/anchor/benchmarks/)
- [B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md)
- [ai-containers vLLM XPU](https://github.com/intel/ai-containers/blob/main/vllm/0.14.1-xpu.md)

OpenVINO:
- [openvino #36061](https://github.com/openvinotoolkit/openvino/issues/36061)
- [benchmarks](https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html)

IPEX / ipex-llm retirement:
- [IPEX install page](https://pytorch-extension.intel.com/)
- [ipex-llm](https://github.com/intel/ipex-llm), [#13324](https://github.com/intel/ipex-llm/issues/13324),
  [#12991](https://github.com/intel/ipex-llm/issues/12991)

Translation layers:
- [SYCLomatic](https://github.com/oneapi-src/SYCLomatic)
- [chipStar](https://github.com/CHIP-SPV/chipStar/releases)

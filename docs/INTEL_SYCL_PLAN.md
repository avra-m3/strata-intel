# Plan: Strata on 2x Intel Arc Pro B60, Linux, several requests at once

A plan, not a feature. Strata has no Intel GPU backend today (CUDA and HIP only; [MULTI_GPU.md](MULTI_GPU.md#limits-for-now)
lists Intel GPUs as not supported), and it serves one request at a time (`serve/server.py`: "One sequence at a time
behind a FIFO"). This page says how to add both, for **one Linux machine** and **one model family**, in stages that
each end in something that can be checked.

Every speed and size on this page that is not quoted from a measurement elsewhere in `docs/` is an **estimate**;
the method is given next to each. Replace them with measurements as each stage lands.

## The machine and the targets

| | | What it means for Strata |
| --- | --- | --- |
| **GPUs** | 2x Intel Arc Pro B60, 24 GB each (Battlemage, Xe2), Level Zero 1.32 | 48 GB of VRAM: with the layer split most of an IQ3 model's experts fit on the cards |
| **CPU** | Ryzen 9 7945HX: 16 Zen 4 cores, 2 CCDs (2 x 32 MB L3), AVX-512 with VNNI, VBMI, BF16 | Strata's AVX-512 i-quant kernels run unchanged; 16 cores instead of the reference PC's 6 |
| **RAM** | 92 GB, one NUMA node (69 GB in use and 23 GB in swap when checked) | the expert arena (43-50 GB) and the streamed KV caches need that memory, see [Before the first run](#before-the-first-run) |
| **OS** | Linux only | no WDDM pinning cap; `xe` kernel driver, Level Zero, oneAPI |
| **Model** | IQ3_XXS first (43 GB of experts), then IQ3_S (50 GB) | the i-quant path: llama.cpp's codebook formats on GPU and CPU |
| **Context** | the model's full 262,144 tokens, 8-bit KV, for every request | no shortening of the context to make room for experts |
| **Requests** | several served at once (target: 4 at full context, more at shorter ones) | new: per-sequence state and batched verify windows in the engine, a scheduler in the server |

## Memory: what goes where

### Per sequence (from measured figures)

- **KV cache, 8-bit:** ~13.7 KB per context token across the 12 attention layers ([DETAILS.md](DETAILS.md#speed-measured),
  KV streaming): **~3.6 GB at 262K**, ~1.8 GB at 128K, ~0.45 GB at 32K.
- **Recurrent state** (36 DeltaNet layers, conv history, n-gram history): ~118 MB (the size of one conversation
  checkpoint, [DETAILS.md](DETAILS.md#using-it)).
- **Draft (MTP) layer KV:** the draft layer attends to the last 32,768 positions only (paper, section 3.7), so it
  stays small at any context.

With the layer split, each card holds the KV and state of **its own layers only** (`SessionState`'s layer-range
carve, `include/strata/core/session.hpp`), so a sequence's ~3.7 GB at 262K is shared between the two cards, about
half each.

**KV streaming** (`--kv-resident N`, engine 0.1.5) keeps only the most-read N cells of each attention layer in VRAM
and the rest in RAM; the attention reads the same values. At N = 32,768 a 262K sequence needs ~0.45 GB of VRAM
(both cards together) and ~3.2 GB of RAM. On this machine that is what makes full context for several requests
affordable: the context lives mostly in RAM, which has room, and the VRAM stays for experts.

### VRAM, per card (estimate)

From the layer-split accounting in [MULTI_GPU.md](MULTI_GPU.md#what-each-card-holds), with a fixed **KV and state
reserve** for concurrency, set aside before the expert cache is sized:

| | Card 0 | Card 1 |
| --- | ---: | ---: |
| VRAM | 24 GB | 24 GB |
| Dense weights (copy) | ~3.5 GB | ~3.5 GB |
| Prompt buffers (2048-token chunks) | ~1.5 GB | ~1.5 GB |
| Draft (MTP) layer | - | ~0.8 GB |
| Driver and runtime reserve (guess) | ~1 GB | ~1 GB |
| **KV and state reserve** (below) | **~3 GB** | **~3 GB** |
| **Left for experts** | **~15 GB** | **~14 GB** |

The 3 GB reserve per card holds, for each concurrent sequence, its resident KV window (32K cells: ~0.23 GB per
card), its recurrent state (~60 MB per card), its verify-window buffers and, on card 1, its draft-layer KV. That is
room for **8 sequences with streamed KV at any context length**, or about **1.5 sequences at 262K with all KV in
VRAM**. With the reserve, the expert cache is ~29 GB: ~67% of IQ3_XXS's 43 GB, ~58% of IQ3_S's 50 GB, filled
hottest first. The reference PC served 72% of routed experts from VRAM with ~3,500 IQ3_XXS experts cached (paper,
findings 4 and 8); here ~16,000 fit. The engine reports the actual hit rate (`--stats`).

**Worth checking early:** whether a later stage needs the *whole* dense set or only its own layers'
(`GpuStage` in `src/program/generate.cpp` loads `pool_bytes` with a `skip` list). Holding only its own layers would
free ~1.5-2 GB per card, about 1,000 more IQ3 experts each, or more KV reserve.

### RAM (estimate)

| | IQ3_XXS | IQ3_S |
| --- | ---: | ---: |
| Expert arena (all experts, pinned) | 43 GB | 50 GB |
| Streamed KV, 4 sequences at 262K | ~13 GB | ~13 GB |
| OS, server, buffers | ~8 GB | ~8 GB |
| **Subtotal** | **~64 GB** | **~71 GB** |
| Left of 92 GB for the n-gram table's page cache and parked conversations | ~28 GB | ~21 GB |

The n-gram table (28.8 GB on the SSD) is read a few rows per token through the page cache; it does not need to fit,
but the more of it stays cached, the fewer SSD reads. 8 sequences at 262K (~26 GB of streamed KV) fit with IQ3_XXS
and leave little with IQ3_S. With a layer split and an IQ pack the whole arena stays in RAM: the low-RAM resident
variant has no layer split yet, and `--mmap-experts` needs a canonical pack.

## Serving several requests at once

Today one sequence owns the engine. The server queues requests (FIFO), and the opt-in conversation parking
(`--conversation-cache-slots`) keeps up to N conversations' states in RAM so alternating requests resume quickly,
but it runs them one after another, and it is rejected together with `--layer-split`. Running them together needs
three things.

### 1. Several sequences resident

Each sequence gets its own `SessionState` (KV, recurrent state, n-gram history, draft-layer KV) on each card, carved
from the KV reserve at start, so a new request does not allocate on the token path (the engine's existing rule:
"ZERO token-path allocations", `session.hpp`). The number of slots is a start option (`--parallel N`), sized by
setup from the reserve and the context.

### 2. One verify window for several sequences (batched decode)

The verify window already carries up to 4 tokens of one sequence through all 48 layers in one pass. The change is to
carry the windows of up to N sequences in the same pass:

- **What is shared:** the dense weights (~3.5 GB, the GPU's main cost per pass) are read once for all sequences.
  That is the whole gain: at batch 1 the GPU reads them for 2.4-3.2 tokens; at batch 4, for 10-13.
- **What is per sequence:** attention (each token reads its own sequence's KV), the DeltaNet recurrence (each
  sequence's tokens run in order on its own state; sequences run side by side), the n-gram rows, the draft layer,
  and acceptance and rollback.
- **The experts:** each token routes to its own 10. The union per layer grows with the batch, so more experts per
  layer, and more misses for the CPU. The multi-token expert kernels (`native_mmvq.cu`'s `launch_multi_n`, up to 8
  tokens per expert; the CPU's multi-token i-quant kernels, 2.0-2.4x at three tokens, paper finding 7) are what keep
  a shared expert cheap. With ~2/3 of the experts on the cards, most of the union is served from VRAM.
- **Drafting at larger batches:** with several sequences in the window, the GPU is busier per pass, so long drafts
  pay less. The draft length per sequence should shrink as the batch grows (measured, not guessed: try 3, 2, 1
  drafts at batch 1, 2, 4, 8).
- **Kernels:** every per-token kernel gains a sequence index (which state, which KV, which position). That is a
  change to their signatures, so **it is designed into the SYCL kernels from stage 2**, even though batching is
  switched on later: porting them single-sequence and changing them again would do the work twice.

### 3. Prompts without stalling everyone else

A 262K-token prompt takes minutes (the reference PC: 4.5 minutes for Q2_0). If it ran in one go, every other request
would stop for that long. Instead the scheduler alternates: one prompt chunk (up to 2,048-8,192 tokens, through the
existing prompt path), then one decode pass for every sequence that is writing, and so on. A long prompt then slows
the others' output by the share of time its chunks take, instead of stopping it. The chunk size sets the trade-off
(larger: faster prompts, longer gaps in the others' output) and is a start option.

The prefix cache (conversation checkpoints) stays per conversation, and with several slots its system-prompt root is
shared: every new chat from the same agent starts reading after the shared system prompt.

### The server

- The FIFO becomes a scheduler with N slots; `/slots` (already an endpoint) lists them as llama.cpp's does, and
  `/metrics` reports per-slot speed.
- The engine's line protocol (stdin/stdout) gains a request id on every line, so the tokens of several requests
  can be told apart.
- Admission: a request that would overflow its slot's context, or arrives when all slots are busy, waits in the
  queue (as now) rather than failing.
- Cancelling one request frees its slot without touching the others.

### What to expect (estimate)

Not measured, and the least certain part of this page. Per request, decode gets slower as the batch grows (bigger
expert union, shorter drafts); in total, it should get faster because the dense weights are read once per pass for
all of them. A guess to check against: at 4 sequences, total output **1.5-2.5x** one sequence's, each sequence at
**40-60%** of its batch-1 speed. Measure batch 1, 2, 4, 8 at 4K and 128K context in stage 7.

## What has to change for Intel

The HIP backend works because HIP accepts CUDA source nearly as is: a 115-line header
(`include/strata/hip_compat/cuda_runtime.h`) renames the runtime and the kernels compile unchanged. SYCL has no such
shortcut: kernels are C++ lambdas or functors, not `__global__` functions launched with `<<<>>>`. So the port has two
halves.

### 1. The runtime calls (host side)

About 70 different CUDA runtime calls are used outside the tests. They map to SYCL / Level Zero as follows; the port
puts them behind one thin layer (`strata::gpu`) so the CUDA and HIP builds keep calling what they call today.

| CUDA (count in `src/`) | SYCL / Level Zero | Note |
| --- | --- | --- |
| `cudaMalloc` / `cudaFree` / `cudaMemcpy*` / `cudaMemset*` (~1,400) | `sycl::malloc_device`, `queue.memcpy`, `queue.memset` | mechanical |
| streams, events (~250) | in-order `sycl::queue`, `sycl::event`, `ext_oneapi_submit_barrier` | `cudaStreamWaitEvent` becomes a barrier on the event |
| `cudaHostAlloc(..., Mapped)` + `cudaHostGetDevicePointer` (~40) | `sycl::malloc_host` (the GPU reads it directly, one pointer for both) | the doorbell, the CPU's results, the hand-off between cards and the streamed KV live here |
| `cudaHostRegister` (13): pinning the 43-50 GB expert arena | no SYCL call; either allocate the arena with `malloc_host` from the start, or Level Zero's `zexDriverImportExternalPointer` | **risk**: how much host memory one process may pin through Level Zero is not known yet |
| CUDA graphs (~130 calls): one captured graph per layer | `sycl_ext_oneapi_graph` (record a queue, then replay) | experimental extension; without it every kernel launch is paid on every pass |
| `cudaLaunchHostFunc` (4) | `host_task` | |
| `cudaMemGetInfo` (25) | `ext::intel::info::device::free_memory` | needs `ZES_ENABLE_SYSMAN=1` |
| `cublasGemmEx` (7): the prompt path's GEMMs | oneMKL `blas::gemm` or oneDNN matmul | |

### 2. The kernels (device side)

50 `.cu` files, ~19,300 lines. Not all of them are on the IQ3 path, and they can be ported in the order the engine
needs them. **Every per-token kernel is ported with a sequence index** (see [batched decode](#2-one-verify-window-for-several-sequences-batched-decode)).

| Order | Files | What for |
| --- | --- | --- |
| 1 | `elementwise.cu`, `quantize_act.cu`, `rope.cu`, `native_rope.cu`, `router_top10.cu`, `native_router.cu`, `sampler.cu` | small, everything depends on them |
| 2 | `iq_kernels.cu` (IQ3_XXS / IQ3_S dot products), `native_mmvq.cu`, `native_moe.cu`, `shared_expert.cu` | the experts at decode: the hot path |
| 3 | `gdn.cu`, `fused_gdn.cu`, `native_gdn*.cu` | the 36 Gated DeltaNet layers (per-sequence state) |
| 4 | `qsa.cu`, `qsa_decode_attn.cu`, `native_qsa*.cu`, `qsa_select.cu`, `kv_q8.cu`, `kv_stream.cu` | the 12 sparse-attention layers, 8-bit KV, KV streaming (per-sequence KV) |
| 5 | `gr.cu`, `fused_gr.cu`, `native_gr_*.cu`, `ple.cu`, `native_ple_postops.cu`, `bf16_gemv.cu`, `native_bf16.cu`, `verify_kernels.cu` | hyper-connections, n-gram embedding, head, verify window |
| 6 | `src/prefill/*.cu` | the prompt path (below) |
| later | `s2_*.cu`, `dequant_s2.cu`, `kv_q4.cu`, `cvec.cu` | Q2_0, 4-bit KV, control vectors: not needed for IQ3 at 8-bit KV |

What needs more than a translation:

- **NVIDIA assembly** (16 lines in `fused_gr.cu`, `qsa_select.cu`, `qsa_prompt_attn.cu`, `native_qsa_score.cu`,
  `verify_kernels.cu`): tensor-core `mma.sync`, `ldmatrix`, `cp.async`, a timer. The HIP build already has plain
  fallbacks for each ("an ordered FP32 fallback" for QSA, [AMD_HIP.md](AMD_HIP.md)); start from those, and move the
  QSA prompt attention to Xe2's matrix units (`joint_matrix`) later. At 262K that attention matters more than at
  short contexts.
- **Warp size.** ~59 shuffles and `warpSize` uses assume 32 lanes (the HIP build relies on wave32 for the same
  reason). Xe2 runs sub-groups of 16 or 32: compile with `[[sycl::reqd_sub_group_size(32)]]` first so the code is
  correct as written, then try 16 on the hottest kernels.
- **`__dp4a`** (the i-quant dot products): SYCL has no portable name; ggml's SYCL backend has a `dp4a` helper that the
  compiler lowers to Xe2's integer dot instruction. `hip_compat/intrinsics.hpp` shows which other CUDA intrinsics
  (`__byte_perm`, packed-byte ops) need the same treatment.
- **The doorbell** (`elementwise.cu:186-222`): the GPU writes the chosen expert ids to host memory that a CPU thread
  spins on, and a GPU kernel waits on a flag the CPU sets. In SYCL that is `atomic_ref` with `memory_scope::system`
  on `malloc_host` memory. A GPU kernel spinning on host memory relies on forward progress the SYCL spec does not
  promise; the HIP build already needed `volatile` stores for this on RDNA4. **Test it first, in isolation**
  (stage 1). With batching, one doorbell per layer carries the union of all sequences' experts.

### The prompt path

`src/prefill/moe_mmq.cu` compiles llama.cpp's CUDA MMQ kernels from the pinned llama.cpp checkout. llama.cpp also has
a SYCL backend (`ggml/src/ggml-sycl`) with IQ3_XXS / IQ3_S dot products and dequantizers. The first version can
dequantize to FP16 and multiply with oneMKL / oneDNN (what Strata did before 0.1.13); a later one can reuse
ggml-sycl's quantized kernels the way the CUDA build reuses ggml-cuda's. With concurrent requests the prompt path
runs in chunks between decode passes, so its buffers are kept (not borrowed from the expert cache and refilled per
request, as on one card): they are in the per-card budget above.

## Stages

Each stage ends in a check. Stages 0 and 2 can be done without the B60s (SYCL also runs on the CPU, through Intel's
OpenCL CPU runtime); the rest need the machine.

| Stage | Work | Done when |
| --- | --- | --- |
| **0. Toolchain** | `STRATA_ENABLE_SYCL` in `CMakeLists.txt` (`cmake/sycl_backend.cmake`, the shape of `hip_backend.cmake`): `icpx -fsycl`, ahead-of-time code for Battlemage (`-fsycl-targets=spir64_gen -Xs "-device bmg-g21"`; check the device name with `ocloc compile --help`). Rejects CUDA + HIP + SYCL in one build. | an empty kernel builds and runs on both B60s and on the CPU device |
| **1. Runtime layer** | `strata::gpu` over CUDA / HIP / SYCL; `device.cu` and `pinned.cu` for SYCL; a standalone doorbell test (GPU writes, CPU spins, CPU writes, GPU waits) and an arena-pinning test (43-50 GB `malloc_host` or imported) | `strata-device` lists both B60s; the doorbell round trip and the pinned arena work, with their times printed |
| **2. Decode kernels** | groups 1-5 above, sequence-indexed, each against the existing parity tests (`src/kernels/*_parity.cpp`, which check against the scalar reference that was itself checked against ggml), plus a two-sequence case per kernel | every parity test on the IQ3 path passes on the CPU device and on a B60, for one and for two sequences |
| **3. One card, one sequence** | `strata generate` on one B60: no graphs, all misses on the CPU (`--pcie-frac 0`), no adaptive swaps | greedy tokens agree with llama.cpp at the rate the CUDA build does (mean KL against llama.cpp ~0.02, the paper's gate); decode tokens/s recorded |
| **4. Two cards** | the layer split (`--gpus 0,1`, `--layer-split auto`, which now also prices the KV reserve): two devices, the hand-off through `malloc_host` memory, no peer-to-peer needed | the hand-off is bit-exact (the existing `--layer-split K --split-device 0` check); hit rate and tokens/s recorded |
| **5. Speed: the overlap** | graphs (`sycl_ext_oneapi_graph`), the doorbell in the hot loop, the copy-engine share of misses (a queue on a Level Zero copy engine), adaptive swaps | each one measured on and off; kept only where it is faster |
| **6. Prompt path and full context** | dequantize + oneMKL GEMM, the PCIe streaming ring, KV streaming, then ggml-sycl's quantized kernels | prompt tokens/s at 4K / 32K / 128K / 262K recorded; needle test 5/5 at 1K-262K |
| **7. Several sequences** | `--parallel N` slots, batched verify windows, per-batch draft length, prompt chunks between decode passes, the server's scheduler and request ids | each sequence's tokens identical to running it alone (greedy, `--pcie-frac 0 --adapt-swaps 0`, the existing reproducibility switches); total and per-request tokens/s at batch 1, 2, 4, 8 at 4K and 128K; time to first token of a short request while a 128K prompt is being read |
| **8. Setup** | `setup.py`: find Intel cards (`xpu-smi`, or `/sys/class/drm/card*/device/vendor` = `0x8086` without it), `--backend sycl`, build with the oneAPI compiler, size the KV reserve and `--parallel` from the RAM and VRAM | `./setup.sh --backend sycl` installs and starts it on this PC |
| **9. Tuning for this PC** | sub-group 16 on the hot kernels, XMX for QSA prompt attention, the CPU pool on the 7945HX (below) | each change measured; this page updated with the numbers |

Stage 7 comes after a fast single-sequence engine on purpose: batching changes what is slow (more experts per
layer, more CPU misses), and it is only possible to tell whether batching helped against a tuned baseline.

## The CPU side on the 7945HX

Nothing to port: Strata's CPU kernels are x86 and this CPU has every instruction they use (AVX-512 VNNI and VBMI).
What differs from the reference PC:

- **16 cores instead of 6.** The pool already pins one worker per physical core (`src/kernels/cpu/pool.cpp`), so it
  would use 15 workers and leave SMT siblings alone. The reference PC's i-quant CPU half was limited by arithmetic
  at ~5 GB/s per core (paper, finding 7); 16 cores could reach ~70 GB/s of arithmetic, so here the RAM's bandwidth
  becomes the limit instead (two channels of DDR5 SO-DIMM: measure it, e.g. with `mbw` or Intel MLC).
- **Two CCDs.** Each has its own 32 MB L3; an expert split across workers on both CCDs costs cross-die traffic.
  Worth trying: whole experts per worker, and workers on one CCD vs both.
- **Batching gives the CPU more work** (the union of several sequences' missed experts) and more shared experts,
  where the multi-token kernels decode a weight once for several tokens. This is where 16 cores pay most.
- **The copy-engine share** (`--pcie-frac`) should be re-measured at each batch size: with few misses at batch 1 the
  best split may differ from the one at batch 8.

## Expected single-sequence speed (estimate, ±50%)

Method: the reference PC's measured 4K IQ3_XXS breakdown, scaled. Its GPU part is set by memory bandwidth (paper,
section 5): the B60's 456 GB/s against the RTX 5070's 672 GB/s makes the GPU's part of a verify window about 1.5x
longer. With most experts in VRAM the CPU's part mostly disappears, so the window becomes GPU-bound.

| Stage reached | IQ3_XXS decode, 4K, one sequence (estimate) | Why |
| --- | ---: | --- |
| 3 (one card, plain) | 15-30 tokens/s | no graphs: ~1,000 kernel launches per window, each paid on the host; ~1/3 of experts on one card |
| 4 (two cards, plain) | 25-45 tokens/s | most experts on the cards, launches still paid |
| 5 (graphs, overlap) | 50-90 tokens/s | GPU-bound windows at 456 GB/s, 2.4-3.2 tokens per window as on the reference PC |

For comparison, the reference PC (RTX 5070 12 GB, Ryzen 5 7600) measured 61.6 tokens/s at 4K and 49.0 at 128K with
IQ3_XXS. These estimates say whether the port is on track, not what it will do.

## Risks

1. **Pinning 43-50 GB of host memory through Level Zero.** If it cannot be done, the GPU cannot read missed experts
   or streamed KV from RAM, and KV streaming (what makes full context for several requests fit) does not work.
   Fallback: KV fully in VRAM at a shorter context or fewer slots, and the CPU computes every miss.
2. **Graphs.** `sycl_ext_oneapi_graph` is experimental; if a layer cannot be recorded, launch overhead caps decode
   (stage 3's estimate). Level Zero immediate command lists keep that overhead lower than the default.
3. **The doorbell.** A GPU kernel that spins on host memory may never see the CPU's write, or stall other work on the
   card. Fallback: let the host enqueue the next layer's expert work after reading the ids (one more round trip per
   layer).
4. **Batching may help less than hoped.** If the union of experts grows close to linearly with the batch, the CPU
   misses grow with it and total speed stays flat. Stage 7 measures it; the slot count is then set by what helps.
5. **Sub-group 32 performance.** Correct first, but Xe2 may run 32-wide sub-groups slower than 16-wide on the hot
   kernels.
6. **Upkeep.** A third backend is a third copy of every kernel to keep in step with the CUDA one, and batched decode
   changes kernels the CUDA and HIP builds share.

## Before the first run

These are on the machine, not in the code:

- **Free the RAM.** 69 GB were in use and 23 GB in swap when checked; the plan needs ~64-71 GB for Strata (above).
- **Allow the pinning:** `ulimit -l unlimited` for the user that runs Strata (or `memlock` in
  `/etc/security/limits.conf`); locking 43-50 GB fails under the usual 8 MB default.
- **Resizable BAR on** in the firmware: Arc cards are much slower without it.
- **The PCIe links:** `sudo lspci -vv -s 03:00.0 | grep LnkSta` and the same for `08:00.0`. The B60 is PCIe 5.0 x8;
  a card on fewer lanes gets a smaller copy-engine share and slower prompts and KV streaming.
- **Driver:** the B60 runs on the kernel's `xe` driver; a recent kernel and Intel's compute runtime (Level Zero,
  OpenCL), and the oneAPI DPC++ compiler.
- **Power:** the 7945HX is a laptop-class chip; its power limits and the CPU governor (`performance`) decide how much
  of its bandwidth the CPU half gets.

## What to collect on the machine first

Before any code, these answer the open questions above:

```sh
xpu-smi discovery -d 0                   # and -d 1: device name, memory, PCIe
sudo lspci -vv -s 03:00.0 | grep -E "LnkCap|LnkSta"
sudo lspci -vv -s 08:00.0 | grep -E "LnkCap|LnkSta"
sycl-ls                                  # the SYCL devices the runtime sees
ocloc compile --help | grep -i bmg       # the AOT device name for the B60
uname -r; modinfo xe | head -3
ulimit -l                                # the pinning limit
sudo dmidecode -t memory | grep -E "Speed|Size" | sort | uniq -c
ps aux --sort=-rss | head -5             # what holds the 69 GB
```

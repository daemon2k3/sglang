# PR description — branch `k31-sm90-fixes`

## Title

```
[Fix] [NVIDIA] attention: Enable fused KDA decode at TP16 and the DFLASH draft KV ring on SM90
```

---

## Motivation

Two SM90-only defects leave substantial memory and performance on the table for
Kimi-K3.1-Pro (`KimiK31ForConditionalGeneration`, DFLASH speculative decoding). Both are
pure Python guard/wiring bugs — the underlying kernels already support the affected
configurations — and both are invisible on Blackwell, which is presumably why they survived.

**1. Fused KDA decode is unreachable at TP16.**
`kimi_k3.py` gates the fused KDA decode kernel on a hardcoded TP8 layout:

```python
seg = 12 * 128  # compiled for H = HV = 12 heads of 128 (TP8)
```

K3.1 has `num_heads=96` over TP, so local heads are `H = HV = 96/TP` → `{12, 6, 3}` for
TP `{8, 16, 32}`. The kernel is generic and already declares exactly that:

```c
// csrc/attention/kda_fused_decode.cuh
// num_heads=96 over TP={8,16,32} -> H=HV={12,6,3} local heads
```

and `kda_fused_decode._SUPPORTED_HEADS = {3, 6, 12}`. At TP16 the guard compares the real
layout `(conv (2304,4), A_log (1,1,6,1), dt_bias (768,))` against the TP8 expectation and
silently disables the fused path for **all 69 KDA layers**.

**2. The DFLASH draft KV ring cannot be used on SM90.**
`SGLANG_DFLASH_DRAFT_KV_RING` (default on) shrinks the draft KV pool from one slot per
*target* token to `window_left + 1 + block_size` rows per request. It is gated on

```python
DFLASH_MLA_ONLY_DRAFT_BACKENDS = ("cutedsl_mla", "trtllm_mla")
```

Both of those funnel into `TllmGenFmhaRunner`, which is Blackwell-only:

```
RuntimeError: Error in function 'TllmGenFmhaRunner' at .../fmhaRunner.cuh:37:
Unsupported architecture
```

(`cutedsl_mla_backend.py` imports `trtllm_mla_backend` and calls
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, so it is the same kernel.)

So on SM90 the ring is always off and the draft pool is sized to the target's full virtual
id space. For a 5-layer, 1024-window draft that is **70× over-allocation**
(1,160,256 rows allocated where 16,576 suffice).

## Modifications

**`python/sglang/srt/models/kimi_k3.py`** — derive the fused-KDA layout instead of assuming TP8:

```python
if w is None or w.ndim != 2 or w.shape[1] != 4: ...
seg = w.shape[0] // 3
H = seg // 128
if (seg != H * 128 or H not in (3, 6, 12) or ... or layer.A_log.numel() != H ...):
```

The guard now validates against the same `{3, 6, 12}` set the kernel advertises, so TP16
(and TP32) engage the fused path.

**`python/sglang/srt/layers/attention/triton_backend.py`** — teach the live window-index path
about the ring. Note the draft does **not** run `TritonMultiStepDraftBackend`: DFLASH forces
`speculative_num_steps=1`, so that class's `attn_backends` list stays empty and it cannot
instantiate. The draft runs a plain `TritonAttnBackend` behind `HybridAttnBackend`, and its
window ids come from `update_sliding_window_buffer` → `pack_read_stream` → `req_to_token`,
i.e. **target-scale ids against a 17,600-row draft pool**. That faults with
`cudaErrorIllegalAddress` on any prefill longer than the ring span. The fix builds ring rows
directly when a ring is installed:

```python
ring = get_draft_kv_ring()
if ring is not None:
    ...
    positions = starts[req_of] + offs
    window_kv_indices[:total] = ring.locs(req_pool_indices[req_of], positions)
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
```

**`python/sglang/srt/speculative/draft_worker_common.py`** — admit `"triton"` to
`DFLASH_MLA_ONLY_DRAFT_BACKENDS` (its window rail is now ring-aware).

## Accuracy Tests

Model: Kimi-K3.1-Pro, TP16/EP16 over 2×H200, DFLASH block 8, triton draft backend,
fp8_e4m3 KV, HiCache L2 + file L3.

| Test | Result |
|---|---|
| `arith` 8-shot | **8/8** |
| GSM8K, n=20, temp 0 | **20/20** |
| Long-context recall (fact at position 0, single needle) | |
| — 21k tokens | ✅ |
| — 60k tokens | ✅ |
| — 100k tokens | ✅ |
| — 128k tokens | ✅ |
| — **256k tokens** | ✅ (250× the ring span) |
| 4 concurrent 60k requests, distinct needles | ✅ all 4 correct — **no cross-request ring contamination** |
| 600-token generation | ✅ |

The concurrent-distinct-needle test is the important one for the ring: it verifies each
request's ring segment is isolated, which is exactly what a wrong index would corrupt.

## Speed Tests and Profiling

Same hardware/config, 8192-token input / 512-token output, unique prompts per run (the
bench uses a random seed to defeat prefix cache — reusing `--seed 42` across runs silently
hits the radix cache and inflates results by up to 2×).

**Fused KDA decode at TP16** (isolated effect):

| cell | before | after | Δ |
|---|---|---|---|
| c1 | 118.74 | 118.44 | ~0 |
| c4 | 198.45 | 197.63 | ~0 |
| c8 | 273.10 | **281.36** | **+3.0%** |
| c16 | 337.70 | 337.59 | ~0 |

Modest but free; the fused path avoids the triton conv+recurrence+norm chain on 69 layers.

**Draft KV ring** (memory, which converts to KV capacity):

| | before | after |
|---|---|---|
| draft KV pool | 3.11 GB | **0.05 GB** |
| target KV pool | 1,160,256 tok | **1,336,960 tok** (**+15.2%**) |

With the ring, the freed 3.06 GB is spent on target KV rather than left idle.

**Concurrency ceiling (measured, not fixed here).** Raising `--max-running-requests` to 32
requires `--cuda-graph-max-bs-decode 32` (see the note below) and 128 mamba slots, which costs
more than the memory freed by the ring:

| config | KV pool | free |
|---|---|---|
| c16, mem 0.95, graph bs 16, mamba 64 | **1,336,960** | 3.80 GB |
| c32, mem 0.95, graph bs 32, mamba 128 | **OOM during graph capture** | — |
| c32, mem 0.94, graph bs 32, mamba 128 | 1,174,656 | 1.14 GB → **OOM under load** |

So c32 costs ~12% KV *and* is not stable: with 1.14 GB free a c32 batch OOMs in prefill
(`Tried to allocate 768.00 MiB ... 352.19 MiB free`). c16 at mem 0.95 is the stable optimum on
2×H200 with this model. The 32-wide graph capture alone costs ~3.9 GB (2.7 GB → 6.6 GB).

**Memory note (separate, not fixed here):** DFLASH sizes its static sampling buffers from
`max(cuda_graph_config.decode.bs)`, while `memory_hook.py` only clamps that value *down* to
`max_running_requests`. The combination `graph_max_bs < max_running_requests` is therefore
unvalidated and crashes with

```
RuntimeError: The size of tensor a (16) must match the size of tensor b (29)
```

in `dflash_worker_v2.py::stage_sampling_params`. A validation (or a fallback to eager when
`bs > graph_max_bs`) would be a worthwhile follow-up; this PR only documents it.

## Checklist

- [x] Format code with pre-commit
- [ ] Add unit tests — **help wanted**: the ring fix is currently covered by the
      end-to-end tests above; a `test/registered/unit/layers/moe`-style unit test for
      `update_sliding_window_buffer` with a ring installed would be a good addition.
- [ ] Update documentation — `SGLANG_DFLASH_DRAFT_KV_RING` deserves a note that it now
      works on SM90 with the triton draft backend.
- [x] Accuracy and speed benchmark results provided above
- [x] Follow the SGLang code style guidance

## Notes for reviewers

- Both changes are guard/wiring only; no kernel code is modified.
- The KDA fix is trivially safe: it accepts a superset of the previous layout, and the
  kernel's own `_SUPPORTED_HEADS` remains the authority.
- The ring fix is the riskier of the two. The failure mode it repairs was a silent
  out-of-bounds read, so the validation above deliberately includes the cases that
  previously faulted (>15k-token prefills) plus concurrent isolation.
- This branch also carries the Kimi-K3.1 humming MXFP4 W4A8 bridge (upstream PR #33884)
  and its kernels-0.1.18 adaptation (#46018) as dependencies; those are not the subject of
  this PR and should land through their own review.

# Attention-output gap analysis — 2026-09-12 (origin/V340_submit, V340 real timeline)

Read-only survey of RESULTS.md (attn rows V198–V350), RULES §10.0t/§10.0u and the code on `origin/V340_submit`
(`src/ops.rs:164-185`, `src/device/sliding/uneven.rs`, `src/device/shared/rmsnorm.rs:340-437`). The furiosa-opt-std
source was not available on this machine, so API claims below (e.g. sub-context `to_vrf` after a vector chain,
`commit_cast` placement) are unverified — check them with a compile probe first. Nothing has been measured yet.

**Shape of the problem** (V346 note): on hardware, contraction1 (cluster 0's tile1 contraction) runs behind the
contraction-store sync (4.7k where the old 24-row tile took 1.5k), so two chains end together: cluster 0's
contraction1 at 40,080 and cluster 1's store sync at 38,966. The tail-rows DM→DM (1.8k) then waits behind the reload.

## 1. Already tried (by axis)

- **Weight load layout / access pattern**: V198 256 B column runs −6.5% adopted; V205 stream 645 B/cycle (diagnostic);
  V196 DMN interleave +6%; V322 one-command 2048/1792 +3.0%; V323 a53 +2.0% · a67 +7.4% (cluster-share line closed);
  V335 stack split +42% (V336 cancelled, V337 designed only); V339 u60 −7.5% load-only; V341 256 B chunk fastest;
  V344 live-slice count no effect (slices 0..127 +43%); V345 k63 −2.9% load-only, floor ≈33.9k.
- **Tile split**: V188 44/16 → V203 88/32 (one tile +2.9k); V266 96/24 −1.25% adopted; V269 100/20 · 92/28 neutral
  (closed); V240 three tiles +887; V304 one tile neutral; V282 shared tile buffer +17%; V224 tile1 on sub compile
  rejected; V340 uneven tiles −1.2% adopted; V342 ua +1.0%, ub +2.4% & FAIL; V346/V347 FAIL (closed).
- **x staging**: V251 direct x load −7.9% adopted; V257 one f8 piece −9.9% (Stage 1 only); V190 x on chip blocked;
  V159 ring broadcast rejected.
- **Contraction knobs**: V240 LaneMode neutral; V248 packet 32 neutral; V262 stream replay +1.8% (qkv); V229 weights in
  TRF held.
- **Store / sync**: V253 split store +3.6%; V260 split store with overlap +10.3%; V254 ring store +5.1%; V195 gathered
  store +3.6–5.7k; V271 256 B-aligned store −1.2% adopted; V284 one cluster +17.5%; V285 order probe (sync independent of
  command order); V290 dummy early store +4.4%; V318 alternating clusters neutral; V281 norm tail on both clusters
  rejected.
- **Norm**: V255 norm in place +4.8%; V263 fused norm (3→2 passes) −1.05% adopted; V274 Main vector pass writing VRF
  directly hangs hardware (PXI-601); V247 hoist loads neutral, sub offload neutral; X5 residual load to head: schedule
  byte-identical; V319 queue order at equal timestamps not source-controllable; V343 local RMS held (inexact).
- **Never tried (no rows)**: `commit_cast`; unpadded HBM tail writes; sqrt folded into sub VRF staging; relabelling the
  tile1 view to share `x_trf`.

## 2. Candidates (exact)

| | Idea | Attacks | Estimate | Main risk | Cheapest check |
|---|---|---|---|---|---|
| **A** | Cluster 0 writes tail rows into HBM instead of the post-reload DM→DM: store into an **unpadded** `HbmTensor<m![H/120, H%120]>` with two disjoint tile stores (contraction rows 0..96 on both clusters; `tails` rows 96..120 on cluster 0), one full reload feeds the norm | contraction1 behind the sync (reload now depends on the tail store) and the DM→DM 40.1–41.9k | **−1.0 to −1.6k** (loses V271 alignment ≈0.5k; command count unchanged) | the lir buffer-size defect (V346, padded layout) may also hit unpadded offset tiles; stores must write disjoint tiles | compile + `--dump-schedule` (contraction1 and tail store before the reload's sync), variant-first accuracy job, 16 paired jobs |
| **B** | Fold +EPS and sqrt into the sub pass that stages the rms VRF (sub vector chain → `to_vrf`), removing the Main rms pass | rms pass 43.1–43.7k, rms VRF 43.8–44.1k | −0.4 to −0.7k (V263 precedent) | `to_vrf` after `vector_final` may not exist on sub; possible V274-style hang; fold placement can hurt (V348 f1u) | compile probe, static dump, one variant-only job |
| **C** | `commit_cast` in the final pass (and contraction passes if it type-checks) | final pass 44.1–45.1k; contraction0 on cluster 1's chain | 0 to −0.5k | cast before `transpose` in contraction passes; invisible to static model | compile final pass first, 16 paired jobs |
| D | 256 B-aligned reducing layout (`m![H # 4096 / 512]`) for scale/weight/residual loads, reload, final store | reload, residual load, store, small loads before tile1 | −0.2 to −0.5k | DM padding garbage must stay out of the sums (V346 NaN); V296 padded-slice ICE; reduce axis innermost (V186) | compile, accuracy job, paired jobs |
| E | Share `x_trf` with contraction1 by relabelling the tile1 view (drop the cluster-0 TRF staging) | 2.8–4.0k (not critical) | ≈0 warm | a TwoClusters pass adds a TU pass on cluster 1; V276 stale-TRF trap | static dump |
| F | Merge the x cast pass into sub TRF staging | 2.0–2.4k | ≈0 (tile0 already issued at 2.0k) | — | compile probe |
| G | Residual load placement | 41.9–43.2k | ≈−0.1k now, ≈−0.3k after A | no source lever (X5, V319) | — |

Rejected at design: spreading norm slices across DMNs (V186, V196); sqrt in the mean-square pass (only Tag/Filter/Output
after an inter-slice reduce); sqrt in the final pass via Stash (needs a sign op, ALU budget); channel scale in the
contraction epilogue (replicated 512-slice scale VRF load at the head, and it cannot follow the inter-slice reduce);
R=88 in plain V340 (cluster 0's chain grows unless A lands first); an attn "remove a store" analogue (the intermediate
store is structural, V284/V255).

## 3. Top 3

1. **A — tail-row HBM store** (−1.0 to −1.6k): the only candidate that attacks both converging chains; gate on a cheap
   compile + static dump for the lir defect on an unpadded layout.
2. **B — sqrt folded into sub VRF staging** (≈−0.5k): V263 precedent; one compile check and one hang check.
3. **C — `commit_cast`** (0 to −0.5k): cheapest to write, fully unmeasured, needs paired jobs.

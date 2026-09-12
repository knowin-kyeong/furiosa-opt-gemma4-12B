# FFN gap analysis — 2026-09-12 (origin/V340_submit ffn = V313_submit ffn)

Produced by a read-only survey of RESULTS.md / RULES.md / the ffn code plus the real span timeline of the production
ffn arm (pod `/root/tk/trim_r11.log`, `decoder_feedforward#1`, window 275.3k). Outcomes added after measurement:

| Candidate | Outcome |
|---|---|
| F1 — sum the f8 pieces inside pass A (vector epilogue) | **V348**: f1d (down x in Lane + in-pass fold) **−1.27%, 16/16 → `V348_submit`** (Arena 25/25 ×2). f1u (up/gate) **+0.77%, 2/16 → rejected**. f1ud −0.75% |
| O1a — one geglu x2 store (re-measure V201) | **V349: −0.43%, 13/16** (adoption candidate); combined with f1d in **V350** (`fo` arm), `V350_submit` staged |
| D2 — f4→bf16 lookup | **Closed**: std 0.6.0 `TableLookup` impls are f4e2m1→f8e4m3, f4e2m1→f8e5m2, f8e5m2→bf16, f8e4m3→bf16 only |
| O1b — one bf16 geglu store, hi/lo split in the down layout | not run |
| T1 — tail scale multiply folded into the post-FF norm | not run (est. −0.5 to −1.5k; needs a new rmsnorm function; rms' = sqrt(g²·ms + eps)/g keeps eps exact) |

---

## 1. Pipeline (origin/V340_submit)

Axes H=3840, L=15360; two clusters × 256 slices.

| Stage | Cluster axis | Slice axis | Element |
|---|---|---|---|
| Head (xsw) | `XCl = Qs/2048` (real 2-cluster) | `XBlocks = [Ns, 1 # 4, H/480]` (64 live/cluster) | `H%480` |
| x replicated | `BothClusters = Dummy2` | `Replicated = Dummy256` | `[Dummy2, H]` f8 |
| up/gate | `UpGateClusters = L/7680` | `UpGateRowsFull = L/30 % 256` | weight `[L%30, H]` f4 57.6 KB; scale `[L%30, H/16]` f8 |
| geglu gathered | same | `[L/480 % 16, 1 # 16]` | `L%480` bf16 |
| x2 in HBM | — | `[L/1920, Dummy2, L%1920]` (8 regions of 3,840 B); inv_s `[L/7680, 1 # 8]` | f8 / f32 |
| down | `DownClusters = H/1920` (rows) | `DownRowsByColumns = [H/60 % 32, L/1920]` | tile `[H%60=r, L%1920]` f4; scale `[H%60, L/16%120]` |
| tail | `Cluster = 1 # 2` | `ReducingSlices = [1 # 32, H/480]` | `H%480` |

Order: residual load (replicated) → fused pre-FF norm (rms weight load, 2 StoVrf, normalize) → `stage_x_hi_lo_full_blocks`
(cast, max², Sub max, 3 pow2 passes, StoVrf s/inv_s, hi, x_hi StoVrf, lo, 2 copy passes into x2, up/gate global scale
loads, erf / half_inv_s² / out_scale passes) → ring-32 `replicate_blocks` + ring-32 erf broadcast → up_w, up_scale,
gate_w, gate_scale, down0, down_scale, down1–3 loads declared → x_trf (Lane) → pass A ×2 (LUT f4→f8, Lane Sequential) →
fold ×2 → pass B 8/8/8/6 rows per matrix → geglu (erf StoVrf, gelu, up×gelu) → `gather_pack_full` (ring-16) →
`stage_geglu_hi_lo_hbm` (max², **ring-256 m_every**, Sub max, pow2, hi/lo, **store x_hi, store x_lo, store inv_s**) →
`broadcast_inv_s_down` (inv_s reload onto 2 slices/cluster, **ring-256 permutation**, StoVrf) → x2 reload (replicated,
32 readers/region) + TRF staging → down tiles 16/16/16/12 (pass A: LUT + Dummy2 time replay; pass B: scale StoVrf, MulF
scale, MulF inv_s, reduces, bf16 commit) → single store (V313) → reload → tail multiply (down_global × out_scale) →
`normalize_add_gate_reduced` → residual store.

Counts: 19 data loads, 6 LUT table loads (StoTab), 5 switch config loads, 5 stores, one ExplicitSync per re-read store
plus the end sync, 5 switches, 6 LUT passes.

## 2. Real timeline (cluster 0, production arm)

| Window (k) | What |
|---|---|
| 0–10.8 | head; up_w issued 10.8k; StoTab#1 9.4–15.5k |
| 10.8–56.8 | up_w (46.0k); up pass A 56.8–78.4k; up fold 78.4–86.0k (7.6k) |
| 56.8–68.9 | two scale DMAs (~12.1k, DMA busy) |
| 68.9–115.0 | gate_w (46.1k), issued when StoTab#2 starts; gate pass A 115.0–136.0k; gate fold 136.0–143.5k |
| 127.1–131.7 | DMA gap 4.6k; down0 issued at StoTab#3 start |
| 148.9–152.0 | gap 3.1k (erf switch → StoVrf f8split.rs:70 → StoVrf mlp.rs:722 before down_scale) |
| 171.6–174.9 | three geglu stores |
| 174.9–190.7 | **syncs 10.3k + 4.0k + 1.5k** |
| 190.7–200.2 | x2 reload 9.5k |
| 200.2–254.3 | down1–3 loads; each later tile load is issued when the previous pass A's StoTab starts (2.3k gaps) |
| 254.3–275.3 | tail 21k: pass A t3 9.1k, pass B, store, sync 1.6k, reload, tail multiply, norm, store |

StoTab spans end when the overlapping Main pass ends, so most of their ~48.6k is Main busy time; the billable part is
that weight loads queue behind table staging.

## 3. Already tried (by axis, abbreviated)

- **up/gate layout/tiles**: V82 12×5 adopted; V181/V182 whole rows −8.9% adopted; V185 4 tiles +8.9k; V216 cyclic rows
  +5.9k; V300 row interleave +27%/+4%; V324 one-command asymmetric +11%; V206/V209 ring 32→4 −16.4k adopted; V199/V200
  merged head stores −11.1k adopted; V293 on-chip x replication −1.8k adopted.
- **pass A / LUT / Lane**: V142 single piece FAIL; V210 Dummy2→Lane blocked (StoVrf replay); V212 bf16 stationary x
  blocked; V223 pass A on Sub blocked; V227/V228 weights in TRF +12.8%/+5.2%; V230 begin_interleaved unreachable; V234
  128-element packet blocked; V235 pieces in Lane + fold neutral; V240 Interleaved rejected; **V301 V235 on V293 path
  −2.3% adopted**; V303/V315 down Lane + fold neutral; "LUT resident" closed at design (table selected by type).
- **geglu / hi-lo / scales / switches**: V264 bulk max pass +1.6%; V272 dropping down global scale FAIL (ε regime);
  **V306 out_scale into tail multiply −1.76% adopted**; V312 +772 static; V328 design-rejected; V331 inv_s without
  ring-256 +1.39%/+3.87%; V332 / m2 blocked (LIR ICE); V333 neutral; V39 blocked.
- **geglu→down hop**: V24a/V168 switch gather +9.5k; V201 one store −628 (2/4, pre-span tooling) → re-measured as V349;
  V250 down cluster axis H→L via switch +13.5% FAIL; V305 down one load +5.0%; V329 design shelved.
- **down tiles/chunks/scales**: V238→V243 16/16/16/12 −724 adopted; V241 +5.9%; V242 +8.6%/compile failure; V314
  +3.9–8.1k; **V260/V261→V313 single output store −1.04% adopted**; chunk-count variants V237 +17.4%, V246 +8.2%, V311
  +67.6k & FAIL, V330 design-rejected.
- **norms/casts**: V263→V267 −0.37% adopted; V274 hangs hardware; V319 no-op; V333 neutral.
- **ordering/head**: V199 adopted; V306 adopted; V308 forced order +0.7%; V309 order screens ≥0.
- No V339/V340-style time split was ever applied to ffn.

## 4. Remaining opportunities (after V348–V350)

- **O1b**: store the geglu output once as bf16 `[L/1920, L%1920]`, reload, and split in the down layout. The global max
  comes from an inter-slice Max over the innermost `L/1920` axis (every down row group holds all 8 chunks), so inv_s
  becomes one value per slice with no switch. Removes syncs #2/#3, the inv_s store/reload/config/permutation switch/
  StoVrf and the m_every ring-256 switch; adds max²/pow2/hi/lo/StoVrf/copy passes on 512 slices × 1920 elements after
  the reload (≈6–8k real). Estimate −2 to −6k. x_hi VRF 7,680 B fits the 8 KB VRF.
- **Down tile count with Lane**: pass A per tile roughly halves with V348; re-sweep 3 tiles (20/20/20) vs 16/16/16/12.
- **T1**: fold down_global × out_scale into the post-FF norm (rms pass computes sqrt(g²·ms + eps)/g), removing the tail
  multiply pass; new rmsnorm function. −0.5 to −1.5k.
- **Uneven up/gate split** (time split): cluster 1's lag builds in up/gate (10.3k hop sync), but cluster 1's rows
  computed on cluster 0 would have to rejoin cluster 1's geglu HBM region — every known route hits a compiler blocker
  (offset HBM tile writes, padded DM buffers). No kernel form known.
- **L-split down** (each cluster multiplies its own geglu half by its own half of the down columns): removes the
  cross-cluster inv_s mapping, but keeps the hop store/sync (per-cluster stores still sync), doubles readers per region
  and makes chunk reads misaligned; estimated neutral or negative.
- **x2 reload copies** (V329 style): each copy adds a store + sync now; negative expected value.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use furiosa_opt_std::prelude::*;

use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops};

mod prng {
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

    pub fn name_hash(name: &str) -> u64 {
        let mut hash = FNV_OFFSET;
        for byte in name.as_bytes() {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    }

    pub fn word(seed: u64, index: usize) -> u64 {
        let mut x = (seed ^ index as u64).wrapping_add(GOLDEN);
        x = (x ^ (x >> 30)).wrapping_mul(MIX_A);
        x = (x ^ (x >> 27)).wrapping_mul(MIX_B);
        x ^ (x >> 31)
    }

    pub fn u01(w: u64) -> f32 {
        ((w >> 40) as u32) as f32 / 16_777_216.0
    }

    pub fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounding = 0x7FFF + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    }

    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn bf16_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        bf16_uniform_range(name, 0, count, lo, hi)
    }

    pub fn bf16_uniform_range(name: &str, offset: usize, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count * 2);
        for index in offset..offset + count {
            push_u16(&mut out, f32_to_bf16_bits(lo + (hi - lo) * u01(word(seed, index))));
        }
        out
    }

    pub fn bf16_signs(name: &str, count: usize, scale: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let magnitude = f32_to_bf16_bits(scale.abs());
        let mut out = Vec::with_capacity(count * 2);
        for index in 0..count {
            let negative = word(seed, index) >> 63 == 1;
            push_u16(&mut out, if negative { magnitude | 0x8000 } else { magnitude });
        }
        out
    }

    pub fn f8_banded(name: &str, count: usize, exp_min: u8, exp_max: u8, signed: bool) -> Vec<u8> {
        assert!(
            exp_min <= exp_max && exp_max <= 14,
            "{name}: exponent band out of range"
        );
        let seed = name_hash(name);
        let span = u64::from(exp_max - exp_min + 1);
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let w = word(seed, index);
            let exponent = exp_min + ((w >> 32) % span) as u8;
            let mantissa = ((w >> 56) & 0x7) as u8;
            let sign = if signed { ((w >> 63) as u8) << 7 } else { 0 };
            out.push(sign | (exponent << 3) | mantissa);
        }
        out
    }

    pub fn f4_nibbles(name: &str, count: usize) -> Vec<u8> {
        assert!(count % 2 == 0, "{name}: f4 element count must be even");
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count / 2);
        for pair in 0..count / 2 {
            let low = ((word(seed, pair * 2) >> 60) & 0xF) as u8;
            let high = ((word(seed, pair * 2 + 1) >> 60) & 0xF) as u8;
            out.push(low | (high << 4));
        }
        out
    }

    pub fn checksum(bytes: &[u8], word_offset: usize) -> u32 {
        let mut total: u32 = 0;
        for (block, chunk) in bytes.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            let index = (word_offset + block) as u32;
            total = total.wrapping_add(u32::from_le_bytes(word).wrapping_mul(index.wrapping_mul(2).wrapping_add(1)));
        }
        total
    }
}

const WEIGHT_EXP: (u8, u8) = (7, 14);
const LOCAL_SCALE_EXP: (u8, u8) = (8, 10);
const ROW_SCALE: (f32, f32) = (0.0, 1e-3);
const UNIT: (f32, f32) = (0.0, 1.0);
const RMS_WEIGHT: (f32, f32) = (0.75, 1.25);

const POS: usize = 137;
const LAYER_SCALAR: f32 = 0.375;

const RAW_GLOBAL_SCALES: [f32; 3] = [9600.0, 9600.0, 12928.0];

struct Fixture {
    expected: HashMap<String, Vec<f32>>,
    checksums: HashMap<String, u32>,
}

const FIXTURE_PATHS: [&str; 2] = ["ref/fixtures.safetensors", "fixtures.safetensors"];

fn fixture_path() -> String {
    if let Ok(path) = std::env::var("GEMMA4_FIXTURE") {
        return path;
    }
    FIXTURE_PATHS
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .unwrap_or(&FIXTURE_PATHS[0])
        .to_string()
}

impl Fixture {
    fn load(path: &str) -> Self {
        let file = File::open(path)
            .unwrap_or_else(|e| panic!("{path}: {e} -- run `python3 scripts/generate_references.py` first"));
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap_or_else(|e| panic!("{path}: {e}"));
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .unwrap_or_else(|e| panic!("{path}: not a safetensors file: {e}"));

        let mut expected = HashMap::new();
        let mut checksums = HashMap::new();
        for (name, view) in tensors.tensors() {
            match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values = view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    expected.insert(name, values);
                }
                safetensors::Dtype::I64 => {
                    let raw = i64::from_le_bytes(view.data()[..8].try_into().unwrap());
                    checksums.insert(name, raw as u32);
                }
                other => panic!("{name}: unexpected fixture dtype {other:?}"),
            }
        }
        Self { expected, checksums }
    }

    fn expect(&self, test: &str, label: &str) -> &[f32] {
        let key = format!("{test}.{label}");
        self.expected
            .get(&key)
            .unwrap_or_else(|| panic!("fixture has no `{key}` -- regenerate with scripts/generate_references.py"))
    }

    fn assert_every_expectation_is_tested(&self) {
        let orphans: Vec<&str> = self
            .expected
            .keys()
            .map(String::as_str)
            .filter(|key| {
                let test = key.split('.').next().unwrap_or(key);
                !PLAN.iter().any(|candidate| candidate.name == test)
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "the fixture has expectations no test reads, so they are silently unchecked: {orphans:?}\n\
             add the matching `Plan` row, `run_plan` arm and shim, or drop the generator"
        );
    }

    fn checksum(&self, test: &str, input: &str) -> u32 {
        let key = format!("{test}.check.{input}");
        *self.checksums.get(&key).unwrap_or_else(|| {
            panic!("fixture has no checksum `{key}` -- regenerate with scripts/generate_references.py")
        })
    }
}

struct Synth<'a> {
    test: &'static str,
    fixture: &'a Fixture,
}

impl<'a> Synth<'a> {
    fn new(test: &'static str, fixture: &'a Fixture) -> Self {
        Self { test, fixture }
    }

    fn seed(&self, name: &str) -> String {
        format!("{}.{}", self.test, name)
    }

    fn verify(&self, name: &str, storage: &[u8]) {
        let expected = self.fixture.checksum(self.test, name);
        let actual = prng::checksum(storage, 0);
        assert_eq!(
            actual, expected,
            "{}.{name}: synthesized input does not match scripts/generate_references.py \
             (checksum {actual:#010x} vs {expected:#010x}) -- fixture_prng.py and \
             this file's prng module have diverged",
            self.test
        );
    }

    async fn upload<D: MaterializableScalar, E: M>(
        &self,
        ctx: &mut Context,
        name: &str,
        storage: Vec<u8>,
    ) -> HbmTensor<D, Chip, E> {
        self.verify(name, &storage);
        HostTensor::<D, E>::from_buf(storage).to_hbm(&mut ctx.pdma).await
    }

    async fn bf16<E: M>(&self, ctx: &mut Context, name: &str, span: (f32, f32)) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_uniform(&self.seed(name), E::SIZE, span.0, span.1);
        self.upload(ctx, name, storage).await
    }

    async fn signs<E: M>(&self, ctx: &mut Context, name: &str, scale: f32) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_signs(&self.seed(name), E::SIZE, scale);
        self.upload(ctx, name, storage).await
    }

    async fn f8<E: M>(
        &self,
        ctx: &mut Context,
        name: &str,
        band: (u8, u8),
        signed: bool,
    ) -> HbmTensor<f8e4m3, Chip, E> {
        let storage = prng::f8_banded(&self.seed(name), E::SIZE, band.0, band.1, signed);
        self.upload(ctx, name, storage).await
    }

    async fn f4<E: M>(&self, ctx: &mut Context, name: &str) -> HbmTensor<f4e2m1, Chip, E> {
        let storage = prng::f4_nibbles(&self.seed(name), E::SIZE);
        self.upload(ctx, name, storage).await
    }

    async fn constant_f32<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<f32, Chip, E> {
        let storage: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_bf16<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<bf16, Chip, E> {
        let storage: Vec<u8> = values
            .iter()
            .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_i32<E: M>(&self, ctx: &mut Context, name: &str, value: i32) -> HbmTensor<i32, Chip, E> {
        self.upload(ctx, name, value.to_le_bytes().to_vec()).await
    }
}

async fn exact_rmsnorm_input(ctx: &mut Context, s: &Synth<'_>) -> HbmTensor<bf16, Chip, m![H]> {
    let signs = prng::bf16_signs(&s.seed("x_signs"), H::SIZE, 1.0);
    s.verify("x_signs", &signs);
    let weights = prng::bf16_uniform(&s.seed("input_rms_weight"), H::SIZE, RMS_WEIGHT.0, RMS_WEIGHT.1);

    let storage: Vec<u8> = signs
        .chunks_exact(2)
        .zip(weights.chunks_exact(2))
        .flat_map(|(sign, weight)| {
            let sign = prng::bf16_bits_to_f32(u16::from_le_bytes(sign.try_into().unwrap()));
            let weight = prng::bf16_bits_to_f32(u16::from_le_bytes(weight.try_into().unwrap()));
            prng::f32_to_bf16_bits(sign / weight).to_le_bytes()
        })
        .collect();
    s.verify("x_exact", &storage);
    HostTensor::<bf16, m![H]>::from_buf(storage).to_hbm(&mut ctx.pdma).await
}

async fn zeros<D: ScalarBytes + MaterializableScalar, E: M>(ctx: &mut Context) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(vec![0u8; E::SIZE * D::BITS / 8])
        .to_hbm(&mut ctx.pdma)
        .await
}

async fn read_bf16<E: M>(ctx: &mut Context, tensor: &HbmTensor<bf16, Chip, E>) -> Vec<f32> {
    let host: HostTensor<bf16, E> = tensor.to_host(&mut ctx.pdma).await;
    host.into_vec().into_iter().map(bf16::to_f32).collect()
}

fn rope_tables(head_dim: usize, theta: f64, partial_rotary_factor: f64, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let angles = (partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize;
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f64; half];
    for (i, slot) in inv_freq.iter_mut().enumerate().take(angles) {
        *slot = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
    }

    let mut cos = vec![0.0f32; head_dim];
    let mut sin = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let angle = pos as f64 * inv_freq[i % half];
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }
    (cos, sin)
}

fn negate_low_half(sin: &[f32]) -> Vec<f32> {
    let half = sin.len() / 2;
    sin.iter()
        .enumerate()
        .map(|(i, v)| if i < half { -v } else { *v })
        .collect()
}

async fn rope_table<D: AxisName>(
    ctx: &mut Context,
    s: &Synth<'_>,
    name: &str,
    values: &[f32],
    pos: usize,
) -> HbmTensor<bf16, Chip, m![E, D]> {
    let row: Vec<u8> = values
        .iter()
        .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
        .collect();
    s.verify(name, &row);

    let mut storage = vec![0u8; E::SIZE * D::SIZE * 2];
    let byte_offset = pos * D::SIZE * 2;
    storage[byte_offset..byte_offset + row.len()].copy_from_slice(&row);
    HostTensor::<bf16, m![E, D]>::from_buf(storage)
        .to_hbm(&mut ctx.pdma)
        .await
}

/// One graded kernel and the exact sequence of launches to make for it.
///
/// `order` has one entry per launch. `""` is the graded kernel; any other string selects an
/// experiment variant inside the shim. When a job compares variants the sequence must be
/// **rotated** so no variant always eats the cold transition -- that confound is what made V219
/// unjudgeable, and V225's Latin square is what fixed it.
struct Plan {
    name: &'static str,
    atol: f32,
    rtol: f32,
    order: &'static [&'static str],
}

const RTOL: f32 = 1e-2;

/// Launches per kernel: one cold (the first) plus REPS-1 warm. V226 raised this from 5 to 13 --
/// the 70 s job cap is host time (per-entry weight re-upload), not device time, and the shims now
/// upload once and launch many times, so repeats are nearly free. Warm run-to-run is +-0.7% on
/// qkv, so 12 warm samples resolve a 1k difference that 4 samples could not.
const REPS: usize = 13;

#[allow(dead_code)]
const BASE: &[&str] = &[""; REPS];

/// V240 sweep, all in one job: ffn production vs the four-tile down stage (`t4`) and the
/// interleaved up/gate lanes (`li`); qkv vs sequential lanes (`ls`); attn_out vs sequential
/// lanes (`ls`) and a three-tile O-weight split (`t3`). Orders are rotated so no variant always
/// eats the cold transition (V225's Latin square).
const FFN_SWEEP: &[&str] = &[""; 3];

const QKV_SWEEP: &[&str] = &[
    "", "sb", "sb", "", "", "sb", "sb", "", "", "sb", "sb", "",
    "", "sb", "sb", "", "", "sb", "sb", "", "", "sb", "sb", "",
];

const ATTN_SWEEP: &[&str] = &[
    "", "hl", "hl", "", "", "hl", "hl", "", "", "hl", "hl", "",
    "", "hl", "hl", "", "", "hl", "hl", "", "", "hl", "hl", "",
];

const PLAN: &[Plan] = &[
    Plan { name: "sliding_attention_output", atol: 0.05, rtol: RTOL, order: ATTN_SWEEP },
    Plan { name: "sliding_project_qkv", atol: 0.04, rtol: RTOL, order: QKV_SWEEP },
    Plan { name: "decoder_feedforward", atol: 0.01, rtol: RTOL, order: FFN_SWEEP },
];

/// Cycle collection for one launch, plus the per-(kernel, variant) sample table.
struct Bench {
    profile: bool,
    settle: Duration,
    collector: Collector,
    samples: RefCell<Vec<(String, u64)>>,
}

impl Bench {
    /// Drop whatever the previous launch left behind; call immediately before `launch`.
    fn arm(&self) {
        if self.profile {
            self.collector.clear();
        }
    }

    /// Spans are decoded off the launch hot path during deferred read-back, so the count keeps
    /// growing for a while after `launch(..).await` returns. Waiting for the count to stop moving
    /// is both safer and far cheaper than V209's flat 500 ms, which cost 500 ms x every entry.
    async fn record(&self, key: &str) {
        if !self.profile {
            return;
        }
        let mut previous = 0usize;
        let mut cycles = None;
        for _ in 0..12 {
            tokio::time::sleep(self.settle).await;
            let observed = self.collector.len();
            if observed > 0 && observed == previous {
                cycles = self.collector.window_cycles();
                break;
            }
            previous = observed;
        }
        match cycles.or_else(|| self.collector.window_cycles()) {
            Some(c) => {
                println!("    {key} cycles={c}");
                self.samples.borrow_mut().push((key.to_string(), c));
            }
            None => println!("    {key} cycles=none observed"),
        }
    }
}

/// `key` for the sample table: the kernel alone for the graded variant, kernel + variant otherwise.
fn key_of(name: &str, variant: &str) -> String {
    if variant.is_empty() { name.to_string() } else { format!("{name} {variant}") }
}

async fn run_plan(ctx: &mut Context, fixture: &Fixture, bench: &Bench, plan: &Plan) -> Vec<(&'static str, Vec<f32>)> {
    match plan.name {
        "sliding_project_qkv" => sliding_project_qkv(ctx, fixture, bench, plan).await,
        "sliding_attention_output" => sliding_attention_output(ctx, fixture, bench, plan).await,
        "decoder_feedforward" => decoder_feedforward(ctx, fixture, bench, plan).await,
        other => panic!("no shim for test `{other}` -- add one in run_plan"),
    }
}

async fn sliding_project_qkv(
    ctx: &mut Context,
    fixture: &Fixture,
    bench: &Bench,
    plan: &Plan,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("sliding_project_qkv", fixture);

    let input_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "input_rms_weight", RMS_WEIGHT).await;
    let x: HbmTensor<bf16, Chip, m![H]> = exact_rmsnorm_input(ctx, &s).await;

    let q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]> = s.f8(ctx, "q_weight", WEIGHT_EXP, true).await;
    let k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(ctx, "k_weight", WEIGHT_EXP, true).await;
    let v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(ctx, "v_weight", WEIGHT_EXP, true).await;
    let q_weight_scale: HbmTensor<bf16, Chip, m![Qs]> = s.bf16(ctx, "q_weight_scale", ROW_SCALE).await;
    let k_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(ctx, "k_weight_scale", ROW_SCALE).await;
    let v_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(ctx, "v_weight_scale", ROW_SCALE).await;
    let q_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "q_rms_weight", UNIT).await;
    let k_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "k_rms_weight", UNIT).await;

    let (cos_values, sin_values) = rope_tables(Ds::SIZE, 10_000.0, 1.0, POS);
    let cos: HbmTensor<bf16, Chip, m![E, Ds]> = rope_table::<Ds>(ctx, &s, "cos", &cos_values, POS).await;
    let sin: HbmTensor<bf16, Chip, m![E, Ds]> =
        rope_table::<Ds>(ctx, &s, "sin", &negate_low_half(&sin_values), POS).await;
    let rope_offset: HbmTensor<i32, Chip, m![1]> =
        s.constant_i32(ctx, "rope_offset", (POS * Ds::SIZE * 2) as i32).await;

    let slot = POS % Ts::SIZE;
    let offset = (slot * Ns::SIZE * Ds::SIZE * 2) as i32;
    let kv_offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(ctx, "kv_offset", offset).await;

    let mut k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
    let mut v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
    let mut q_out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = zeros(ctx).await;

    // Nothing here is re-uploaded between launches: qkv is idempotent (every launch writes the
    // same values into the same k/v-cache slot and into q_out), so the weights, tables and caches
    // are built once and the loop is pure launch + measure. The read-back is 8.4 MB, so it runs
    // once, after the first launch.
    let mut outputs: Vec<(&'static str, Vec<f32>)> = Vec::new();
    for (i, variant) in plan.order.iter().enumerate() {
        bench.arm();
        match *variant {
            "" => {
                launch(
                    ops::sliding_project_qkv,
                    (
                        ctx,
                        &x,
                        &q_weight,
                        &k_weight,
                        &v_weight,
                        &q_weight_scale,
                        &k_weight_scale,
                        &v_weight_scale,
                        &input_rms_weight,
                        &q_rms_weight,
                        &k_rms_weight,
                        &kv_offset,
                        &rope_offset,
                        &cos,
                        &sin,
                        &mut k_cache,
                        &mut v_cache,
                        &mut q_out,
                    ),
                )
                .await;
            }
            "ls" => {
                launch(
                    ops::sliding_project_qkv_seq,
                    (
                        ctx,
                        &x,
                        &q_weight,
                        &k_weight,
                        &v_weight,
                        &q_weight_scale,
                        &k_weight_scale,
                        &v_weight_scale,
                        &input_rms_weight,
                        &q_rms_weight,
                        &k_rms_weight,
                        &kv_offset,
                        &rope_offset,
                        &cos,
                        &sin,
                        &mut k_cache,
                        &mut v_cache,
                        &mut q_out,
                    ),
                )
                .await;
            }
            "sb" => {
                launch(
                    ops::sliding_project_qkv_sub,
                    (
                        ctx,
                        &x,
                        &q_weight,
                        &k_weight,
                        &v_weight,
                        &q_weight_scale,
                        &k_weight_scale,
                        &v_weight_scale,
                        &input_rms_weight,
                        &q_rms_weight,
                        &k_rms_weight,
                        &kv_offset,
                        &rope_offset,
                        &cos,
                        &sin,
                        &mut k_cache,
                        &mut v_cache,
                        &mut q_out,
                    ),
                )
                .await;
            }
            "ls" => {
                launch(
                    ops::sliding_project_qkv_seq,
                    (
                        ctx,
                        &x,
                        &q_weight,
                        &k_weight,
                        &v_weight,
                        &q_weight_scale,
                        &k_weight_scale,
                        &v_weight_scale,
                        &input_rms_weight,
                        &q_rms_weight,
                        &k_rms_weight,
                        &kv_offset,
                        &rope_offset,
                        &cos,
                        &sin,
                        &mut k_cache,
                        &mut v_cache,
                        &mut q_out,
                    ),
                )
                .await;
            }
            other => panic!("no variant `{other}` for sliding_project_qkv"),
        }
        bench.record(&key_of(plan.name, variant)).await;

        if plan.order[..i].iter().all(|seen| seen != variant) {
            let width = Ns::SIZE * Ds::SIZE;
            let k = read_bf16(ctx, &k_cache).await[slot * width..(slot + 1) * width].to_vec();
            let v = read_bf16(ctx, &v_cache).await[slot * width..(slot + 1) * width].to_vec();
            outputs.push(("expected.q", read_bf16(ctx, &q_out).await));
            outputs.push(("expected.k", k));
            outputs.push(("expected.v", v));
        }
    }
    outputs
}

async fn sliding_attention_output(
    ctx: &mut Context,
    fixture: &Fixture,
    bench: &Bench,
    plan: &Plan,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("sliding_attention_output", fixture);
    let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.signs(ctx, "x", 1.0).await;
    let post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "post_attn_rms_weight", UNIT).await;
    let o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]> = s.f8(ctx, "o_weight", WEIGHT_EXP, true).await;
    let o_weight_scale: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "o_weight_scale", ROW_SCALE).await;
    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;

    // `residual` is read-modify-write, so it is the one tensor that has to be restored before each
    // launch (7.7 KB); everything else is read-only and was uploaded once above.
    let mut outputs: Vec<(&'static str, Vec<f32>)> = Vec::new();
    for (i, variant) in plan.order.iter().enumerate() {
        if i > 0 {
            residual = s.bf16(ctx, "residual", UNIT).await;
        }
        bench.arm();
        match *variant {
            "" => {
                launch(
                    ops::sliding_attention_output,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            "ls" => {
                launch(
                    ops::sliding_attention_output_seq,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            "t3" => {
                launch(
                    ops::sliding_attention_output_t3,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            "hl" => {
                launch(
                    ops::sliding_attention_output_hoist,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            "ls" => {
                launch(
                    ops::sliding_attention_output_seq,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            "t3" => {
                launch(
                    ops::sliding_attention_output_t3,
                    (
                        ctx,
                        &x,
                        &post_attn_rms_weight,
                        &o_weight,
                        &o_weight_scale,
                        &mut residual,
                    ),
                )
                .await;
            }
            other => panic!("no variant `{other}` for sliding_attention_output"),
        }
        bench.record(&key_of(plan.name, variant)).await;

        if i == 0 {
            outputs = vec![("expected", read_bf16(ctx, &residual).await)];
        }
    }
    outputs
}

async fn decoder_feedforward(
    ctx: &mut Context,
    fixture: &Fixture,
    bench: &Bench,
    plan: &Plan,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("decoder_feedforward", fixture);

    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;
    let pre_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "pre_ff_rms_weight", UNIT).await;
    let post_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "post_ff_rms_weight", UNIT).await;

    let up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(ctx, "up_weight_packed").await;
    let gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(ctx, "gate_weight_packed").await;
    let down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]> = s.f4(ctx, "down_weight_packed").await;
    let up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(ctx, "up_weight_scale", LOCAL_SCALE_EXP, false).await;
    let gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(ctx, "gate_weight_scale", LOCAL_SCALE_EXP, false).await;
    let down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]> =
        s.f8(ctx, "down_weight_scale", LOCAL_SCALE_EXP, false).await;

    let up_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "up_global_scale", &[1.0 / RAW_GLOBAL_SCALES[0]])
        .await;
    let gate_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "gate_global_scale", &[1.0 / RAW_GLOBAL_SCALES[1]])
        .await;
    let down_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "down_global_scale", &[1.0 / RAW_GLOBAL_SCALES[2]])
        .await;

    let layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]> = s.constant_bf16(ctx, "layer_scalar", &[LAYER_SCALAR; 8]).await;

    // 99.5 MB of packed weights and scales, uploaded once. Only `residual` (7.7 KB) is restored
    // per launch -- without that the second launch compounds the first one's output and FAILs.
    let mut outputs: Vec<(&'static str, Vec<f32>)> = Vec::new();
    for (i, variant) in plan.order.iter().enumerate() {
        if i > 0 {
            residual = s.bf16(ctx, "residual", UNIT).await;
        }
        bench.arm();
        match *variant {
            "" => {
                launch(
                    ops::decoder_feedforward,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "s1" => {
                launch(
                    ops::decoder_feedforward_s1,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "c4" => {
                launch(
                    ops::decoder_feedforward_v246,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "s1" => {
                launch(
                    ops::decoder_feedforward_s1,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d2" => {
                launch(
                    ops::decoder_feedforward_v241,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "d4" => {
                launch(
                    ops::decoder_feedforward_v237,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            "t4" => {
                launch(
                    ops::decoder_feedforward_v238,
                    (
                        ctx,
                        &mut residual,
                        &pre_ff_rms_weight,
                        &up_weight_packed,
                        &gate_weight_packed,
                        &down_weight_packed,
                        &up_weight_scale,
                        &gate_weight_scale,
                        &down_weight_scale,
                        &up_global_scale,
                        &gate_global_scale,
                        &down_global_scale,
                        &post_ff_rms_weight,
                        &layer_scalar,
                    ),
                )
                .await;
            }
            other => panic!("no variant `{other}` for decoder_feedforward"),
        }
        bench.record(&key_of(plan.name, variant)).await;

        // Compare the first launch of each distinct variant, not just the first launch overall:
        // a variant that is fast but wrong must not pass unnoticed.
        if plan.order[..i].iter().all(|seen| seen != variant) {
            outputs.push(("expected", read_bf16(ctx, &residual).await));
        }
    }
    outputs
}

fn compare(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> bool {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: shape mismatch ({} expected vs {} from device)",
        expected.len(),
        actual.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_diff = 0.0f64;
    let mut within = 0usize;

    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if !got.is_finite() {
            println!("[{label:34}] FAIL -- non-finite device output at {index}");
            return false;
        }
        let diff = (want - got).abs();
        if diff <= atol + rtol * want.abs() {
            within += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
        sum_diff += f64::from(diff);
    }

    let count = expected.len();
    let ok = within == count;
    let relative = if expected[max_index].abs() > 1e-12 {
        max_diff / expected[max_index].abs() * 100.0
    } else {
        0.0
    };
    println!(
        "[{label:34}] max|Δ|={max_diff:9.5} ({relative:7.2}% of expected)  mean|Δ|={:9.6}  \
         within tol={:6.2}%  -> {}",
        sum_diff / count as f64,
        within as f32 / count as f32 * 100.0,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// --- on-device cycle collection (only armed when TUC_PROFILE_LEVEL is set) ---

const TRACING_TARGET_NPU: &str = "span::npu";

#[derive(Clone, Copy)]
struct Span {
    begin: u64,
    end: u64,
}

/// A minimal `tracing::Subscriber`: all we need is to see each `span::npu` span's fields
/// as it's created, not the full `Layer`/`Registry` machinery `tracing-subscriber` offers.
#[derive(Clone, Default)]
struct Collector {
    spans: Arc<Mutex<Vec<Span>>>,
    next_id: Arc<AtomicU64>,
}

impl Collector {
    fn clear(&self) {
        self.spans.lock().unwrap().clear();
    }

    /// How many spans have been decoded so far. `Bench::record` polls this: the count going
    /// still is the signal that deferred read-back has finished for this launch.
    fn len(&self) -> usize {
        self.spans.lock().unwrap().len()
    }

    /// Real total cycles for whatever ran since the last `clear`: the union of every
    /// span observed (min begin .. max end).
    fn window_cycles(&self) -> Option<u64> {
        let spans = self.spans.lock().unwrap();
        let begin = spans.iter().map(|s| s.begin).min()?;
        let end = spans.iter().map(|s| s.end).max()?;
        Some(end.saturating_sub(begin))
    }
}

#[derive(Default)]
struct FieldExtractor {
    begin: Option<u64>,
    end: Option<u64>,
}

impl tracing::field::Visit for FieldExtractor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "begin_cycle" => self.begin = Some(value),
            "end_cycle" => self.end = Some(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl tracing::Subscriber for Collector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == TRACING_TARGET_NPU
    }

    fn new_span(&self, attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        if attrs.metadata().target() == TRACING_TARGET_NPU {
            let mut extractor = FieldExtractor::default();
            attrs.record(&mut extractor);
            if let (Some(begin), Some(end)) = (extractor.begin, extractor.end) {
                self.spans.lock().unwrap().push(Span { begin, end });
            }
        }
        // 0 is reserved by `span::Id`; spans aren't tracked individually here, so the
        // id only needs to be unique and non-zero.
        tracing::span::Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

fn profiling_enabled() -> bool {
    let level = std::env::var("TUC_PROFILE_LEVEL").unwrap_or_default().to_ascii_lowercase();
    matches!(level.as_str(), "info" | "debug" | "trace")
}

/// Poll interval for `Bench::record`, not a flat wait. V209 slept a flat 500 ms per entry, which
/// is 500 ms x every launch of dead time -- the single largest fixed cost in a job once the weight
/// uploads are hoisted. The Arena entrypoint cannot inject `GEMMA4_PROFILE_SETTLE_MS`, so the
/// default is what a submitted job gets.
fn settle() -> Duration {
    let ms = std::env::var("GEMMA4_PROFILE_SETTLE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100u64);
    Duration::from_millis(ms)
}

#[tokio::main]
async fn main() {
    let fixture = Fixture::load(&fixture_path());
    fixture.assert_every_expectation_is_tested();
    let mut ctx = Context::acquire();

    let profile = profiling_enabled();
    let collector = Collector::default();
    if profile {
        tracing::subscriber::set_global_default(collector.clone()).expect("set global tracing subscriber");
    }
    let bench = Bench {
        profile,
        settle: settle(),
        collector,
        samples: RefCell::new(Vec::new()),
    };

    let launches: usize = PLAN.iter().map(|p| p.order.len()).sum();
    println!(
        "NPU kernel tests -- {} kernels, {launches} launches against a precomputed reference{}\n",
        PLAN.len(),
        if profile { ", with on-device cycle counts" } else { "" }
    );

    let started = Instant::now();
    let mut failures = Vec::new();
    for plan in PLAN {
        println!("==> {} ({} launches)", plan.name, plan.order.len());
        let kernel_started = Instant::now();

        let outputs = run_plan(&mut ctx, &fixture, &bench, plan).await;

        assert!(!outputs.is_empty(), "{}: shim produced no outputs to compare", plan.name);
        let mut ok = true;
        for (label, actual) in &outputs {
            let display = if outputs.len() == 1 {
                plan.name.to_string()
            } else {
                format!("{} {}", plan.name, label.trim_start_matches("expected."))
            };
            ok &= compare(&display, fixture.expect(plan.name, label), actual, plan.atol, plan.rtol);
        }
        println!("    elapsed={:.1}s", kernel_started.elapsed().as_secs_f64());
        println!();

        if !ok {
            failures.push(plan.name);
        }
    }

    // Per (kernel, variant): the first launch is cold (that is what the grader draws), median and
    // stdev over the warm remainder. Only compare variants inside one job, adjacent launches.
    let samples = bench.samples.borrow();
    if !samples.is_empty() {
        println!("summary (cold = first launch; median/stdev over the later launches)");
        let mut keys: Vec<String> = samples.iter().map(|(k, _)| k.clone()).collect();
        let mut seen = std::collections::HashSet::new();
        keys.retain(|k| seen.insert(k.clone()));
        for key in keys {
            let values: Vec<u64> = samples.iter().filter(|(k, _)| *k == key).map(|(_, c)| *c).collect();
            let cold = values[0];
            let mut warm: Vec<u64> = values[1..].to_vec();
            warm.sort_unstable();
            let (median, stdev) = if warm.is_empty() {
                (cold as f64, 0.0)
            } else {
                let n = warm.len();
                let median = if n % 2 == 1 {
                    warm[n / 2] as f64
                } else {
                    (warm[n / 2 - 1] + warm[n / 2]) as f64 / 2.0
                };
                let mean = warm.iter().sum::<u64>() as f64 / n as f64;
                let variance = warm.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                (median, variance.sqrt())
            };
            let spread = if median > 0.0 { stdev / median * 100.0 } else { 0.0 };
            println!(
                "    {key:34} n={:2} cold={cold} median={median:.0} stdev={stdev:.0} ({spread:.2}%) min={} max={}",
                values.len(),
                warm.first().copied().unwrap_or(cold),
                warm.last().copied().unwrap_or(cold),
            );
        }
    }
    drop(samples);

    println!();
    println!("total elapsed={:.1}s", started.elapsed().as_secs_f64());
    if failures.is_empty() {
        println!("all kernel tests passed ({} kernels, {launches} launches)", PLAN.len());
    } else {
        println!(
            "{} of {} kernels failed: {}",
            failures.len(),
            PLAN.len(),
            failures.join(", ")
        );
    }
    std::process::exit(i32::from(!failures.is_empty()));
}

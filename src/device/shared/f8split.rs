//! Carrying a bf16 vector as two f8e4m3 pieces so that a projection can contract it as f8 x f8
//! against an f8 (or f4-decoded) weight stream with no lookup pass: x is scaled by a power of
//! two s chosen from max x^2 (max |x s| in (64, 128]), then hi = f8(x s), lo = f8(x s - hi); their
//! sum is bf16(x) s exactly down to max |x| / 4096 (each piece carries 4 significant bits), and the
//! consumer either folds 1/s in or, for an RMSNorm, needs nothing (scale-invariant up to eps).
//! Macros because the layouts differ per kernel; see mlp.rs and projection.rs for the uses.

/// From `m` (a max of squares, one `1 # 8` packet per slice) to `s = 2^floor(log2(128 / sqrt m))`
/// and `1 / s`: a power of two that brings max |x| into (64, 128], so that x * s is exact in bf16
/// and its two f8e4m3 pieces keep 8 significant bits down to max |x| / 4096.
#[macro_export]
macro_rules! pow2_scale_fns {
    ($name:ident, $cl:ty, $sl:ty) => {
        fn $name(
            ctx: &mut Context,
            m: &DmTensor<f32, Chip, $cl, $sl, m![1 # 8]>,
        ) -> (DmTensor<f32, Chip, $cl, $sl, m![1 # 8]>, DmTensor<f32, Chip, $cl, $sl, m![1 # 8]>) {
            let t: DmTensor<f32, Chip, $cl, $sl, m![1 # 8]> = ctx
                .sub
                .begin(m.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_fp_div_with_mode(BinaryArgMode::Mode10, 128f32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            // Keeping only the exponent field rounds t down to a power of two.
            let s: DmTensor<f32, Chip, $cl, $sl, m![1 # 8]> = ctx
                .sub
                .begin(t.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_logic(LogicBinaryOpF32::BitAnd, f32::INFINITY)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let inv_s: DmTensor<f32, Chip, $cl, $sl, m![1 # 8]> = ctx
                .sub
                .begin(s.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1f32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            (s, inv_s)
        }
    };
}

/// A `1 # 8` f32 packet staged into the VRF (any layout).
#[macro_export]
macro_rules! stage_packet_fns {
    ($name:ident, $cl:ty, $sl:ty) => {
        fn $name(
            ctx: &mut Context,
            v: &DmTensor<f32, Chip, $cl, $sl, m![1 # 8]>,
        ) -> VrfTensor<f32, Chip, $cl, $sl, m![1 # 8]> {
            ctx.sub
                .begin(v.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf()
        }
    };
}

/// Max of x^2 over one slice's elements of a bf16 vector, as a `1 # 8` packet.
#[macro_export]
macro_rules! max_square_fns {
    ($name:ident, $cl:ty, $sl:ty, $ax:ident, $n:literal, $n16:literal, $n8:literal, $n4:literal) => {
        fn $name(
            ctx: &mut Context,
            x: &DmTensor<bf16, Chip, $cl, $sl, m![$ax % $n]>,
        ) -> DmTensor<f32, Chip, $cl, $sl, m![1 # 8]> {
            ctx.sub
                .begin(x.view())
                .fetch::<m![$ax / 16 % $n16], m![$ax % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$ax / 4 % $n4], m![$ax % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<$ax, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit()
        }
    };
}

/// `hi_lo_fns!` with the scale as an immediate: for a consumer whose input is bounded a priori
/// (the attention output, |x| <= sqrt(Ds) = 16) no scale has to be measured.
#[macro_export]
macro_rules! hi_lo_const_fns {
    ($name:ident, $cl:ty, $sl:ty, $ax:ident, $n:literal, $n32:literal, $n16:literal, $n8:literal, $n4:literal) => {
        fn $name(
            ctx: &mut Context,
            x: &DmTensor<bf16, Chip, $cl, $sl, m![$ax % $n]>,
            s: f32,
        ) -> (
            DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]>,
            DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]>,
        ) {
            let x_hi: DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x.view())
                .fetch::<m![$ax / 16 % $n16], m![$ax % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$ax / 4 % $n4], m![$ax % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s)
                .vector_widen_concat::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_final()
                .cast::<f8e4m3, m![$ax % 8 # 32]>()
                .commit_trim::<m![$ax % 8]>()
                .commit();

            let x_hi_vrf: VrfTensor<f32, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x_hi.view())
                .fetch::<m![$ax / 32 % $n32], m![$ax % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .to_vrf();

            let x_lo: DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x.view())
                .fetch::<m![$ax / 16 % $n16], m![$ax % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$ax / 4 % $n4], m![$ax % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s)
                .vector_fp_binary(FpBinaryOp::SubF, &x_hi_vrf)
                .vector_widen_concat::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_final()
                .cast::<f8e4m3, m![$ax % 8 # 32]>()
                .commit_trim::<m![$ax % 8]>()
                .commit();
            (x_hi, x_lo)
        }
    };
}

/// The two f8 pieces of a scaled vector: `hi = f8(x * s)` and `lo = f8(x * s - hi)`. Their sum is
/// bf16(x) * s exactly (x has 8 significant bits, each f8e4m3 piece carries 4, and s is a power of
/// two), so an f8 x f8 contraction against both reproduces the bf16 x f8 one, up to 1/s.
#[macro_export]
macro_rules! hi_lo_fns {
    ($name:ident, $cl:ty, $sl:ty, $ax:ident, $n:literal, $n32:literal, $n16:literal, $n8:literal, $n4:literal) => {
        fn $name(
            ctx: &mut Context,
            x: &DmTensor<bf16, Chip, $cl, $sl, m![$ax % $n]>,
            s_vrf: &VrfTensor<f32, Chip, $cl, $sl, m![1 # 8]>,
        ) -> (
            DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]>,
            DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]>,
        ) {
            let x_hi: DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x.view())
                .fetch::<m![$ax / 16 % $n16], m![$ax % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$ax / 4 % $n4], m![$ax % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s_vrf)
                .vector_widen_concat::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_final()
                .cast::<f8e4m3, m![$ax % 8 # 32]>()
                .commit_trim::<m![$ax % 8]>()
                .commit();

            let x_hi_vrf: VrfTensor<f32, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x_hi.view())
                .fetch::<m![$ax / 32 % $n32], m![$ax % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .to_vrf();

            let x_lo: DmTensor<f8e4m3, Chip, $cl, $sl, m![$ax % $n]> = ctx
                .sub
                .begin(x.view())
                .fetch::<m![$ax / 16 % $n16], m![$ax % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$ax / 4 % $n4], m![$ax % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s_vrf)
                .vector_fp_binary(FpBinaryOp::SubF, &x_hi_vrf)
                .vector_widen_concat::<m![$ax / 8 % $n8], m![$ax % 8]>()
                .vector_final()
                .cast::<f8e4m3, m![$ax % 8 # 32]>()
                .commit_trim::<m![$ax % 8]>()
                .commit();
            (x_hi, x_lo)
        }
    };
}

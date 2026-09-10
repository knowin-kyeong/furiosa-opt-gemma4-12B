
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub(crate) type Cluster = m![1 # 2];
pub(crate) type Slice = m![1 # 256];

pub(crate) type Replicated = m![Dummy256];
/// Cluster mapping that replicates a tensor onto both clusters.
pub(crate) type BothClusters = m![Dummy2];
/// One KV head per slice within a ring of eight slices (the layout RoPE works in).
pub(crate) type HeadSlices = m![1 # 32, Ns];
/// Four KV heads per cluster, one per live slice, after a ring-64 gather of the projection rows.
pub(crate) type HeadClusters = m![Ns / 4];
pub(crate) type HeadSlicesPerCluster = m![Ns % 4, 1 # 64];

pub(crate) fn broadcast_hidden(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![H]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    unsafe { x.reshape() }
}

pub(crate) fn broadcast_sliding_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Qs]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qs]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();

    unsafe { x.reshape() }
}

pub(crate) fn broadcast_full_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Qf]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Qf]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qf]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qf / 16], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();

    unsafe { x.reshape() }
}

/// V217: one channel per slice. Slice `d` holds channel `d` of every query/key/value head its
/// cluster owns, so a head's 256 channels lie across the 256 slices instead of on one of them.
/// That makes the head RMSNorm an inter-slice reduce instead of a ring-64 gather, turns the RoPE
/// tables and the per-channel scales into one value per slice instead of 256, and lets the weight
/// arrive as eight 3,840-byte 256-byte-aligned runs (V215: 514 B/cycle against 465).
pub(crate) type ChannelSlices = m![Ds];

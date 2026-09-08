## Overview

Participants start from a baseline implementation of Gemma 4 12B written for [RNGD](https://furiosa.ai/rngd) with the furiosa-opt toolchain and improve its inference performance. Participants do not need their own RNGD hardware: submitted programs are run on FuriosaAI's RNGD servers through FuriosaAI Arena.

The competition is held in two rounds. The *Kernel Optimization Round* targets three decoder-layer kernels of the model — the QKV projection, the attention output projection and the feed-forward block. Participants with the highest results advance to the *Model Optimization Round*, in which the full model is optimized end to end. The top three teams will have the opportunity to present their work at the MOA Workshop in Athens.

In both rounds, submissions must pass a correctness check and are then ranked by performance. Detailed rules and evaluation criteria are specified in the [baseline repository](https://github.com/HoseongLee/furiosa-opt-gemma4-12B).

## Getting started

Tutorial session

We are preparing a tutorial session for participants. The date and details will be announced here soon.

1. RegisterFill out the [registration form](https://docs.google.com/forms/d/e/1FAIpQLScOBksLrh4AW5ApaWLEEAdY1pXb4nwV6N1_DFuDWO8kCRc_4A/viewform) by September 15, 2026.
2. Set up the baselineClone the [baseline repository](https://github.com/HoseongLee/furiosa-opt-gemma4-12B), a working implementation of Gemma 4 12B for RNGD. Its README is the competition guide: it covers the toolchain setup, the three target kernels, the permitted changes and the grading test. OPTIMIZATION.md walks through the optimization workflow. The [furiosa-opt documentation](https://developer.furiosa.ai/furiosa-opt/book/) is the companion programming guide, covering everything from the vISA programming model to scheduling and tuning.
3. Optimize and testImprove the target kernels and check your work on real hardware as you go: the `rngd` command-line client runs your code on [FuriosaAI Arena](https://arena.furiosa.ai/), the RNGD evaluation server. Arena access is provided to registered participants.
4. SubmitWhen you are ready, submit your entry as described in the baseline repository's README. Submissions that pass the correctness check are ranked by performance on the [leaderboard](https://micro2026-moa.github.io/leaderboard.html), and top performers in the Kernel Optimization Round advance to the Model Optimization Round.
Technical questions are best asked on the [FuriosaAI forums](https://forums.furiosa.ai/), where answers are shared with all participants.

## Important dates

ItemDateRegistration PeriodSep 1 – Sep 15, 2026Kernel Optimization RoundSep 1 – Sep 25, 2026Finalists AnnouncementSep 30, 2026Model Optimization RoundOct 1 – Oct 25, 2026Award Ceremony (time and venue TBA)Nov 1, 2026
The award ceremony takes place at the MOA Workshop, held on November 1, 2026 at MICRO 2026 in Athens, Greece.

## Resources

- [Baseline repositoryGemma 4 12B baseline for RNGD. The README also serves as the competition guide. It specifies the target kernels, permitted changes and correctness tolerances, and explains how to set up the toolchain, optimize a kernel and run the grading test.github.com/HoseongLee/furiosa-opt-gemma4-12B](https://github.com/HoseongLee/furiosa-opt-gemma4-12B)
- [furiosa-opt documentationProgramming guide for the furiosa-opt toolchain in which the baseline is written. Covers setup, the vISA programming model, scheduling, and development tools such as the schedule viewer.developer.furiosa.ai/furiosa-opt/book](https://developer.furiosa.ai/furiosa-opt/book/)
- [FuriosaAI ArenaJob scheduler for FuriosaAI's shared RNGD servers, on which submitted programs are executed. Jobs are submitted with the rngd command-line client; setup instructions are in the baseline repository. Access is provided to registered participants.](https://arena.furiosa.ai/)

## Objectives
Round 1. Kernel Optimization
Each board shows the best valid result from each team, ranked by performance. Score is the geometric mean of its per-kernel speedups over the baseline. (QKV projection, attention output projection, and decoder feed-forward block cycles) The top performers advance to Round 2.
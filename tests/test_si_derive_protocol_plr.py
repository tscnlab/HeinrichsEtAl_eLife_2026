"""Tests for the registered FLUX-normalised pupil outcome.

Scope is deliberately narrow.  The criterion for including a test is whether
the failure would be *invisible in the output*:

- The order of operations (max per trial, then average) differs from the
  reverse by only 0.06-0.15 mm on real data and both land near 1 mm, so a swap
  produces entirely plausible numbers.
- A groupby that drops ``Test`` silently pools test repetitions and still
  returns sensible amplitudes.
- Applying the six-block rule before pairing rather than after changes which
  participants survive, without any error.

Everything else in the derivation fails loudly or cannot occur on this dataset
(zero denominators, FLUX leaking into the modelled set), and is covered by the
assertions in ``si_derive_protocol_plr`` rather than by a test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from civibeplr.si_derive_protocol_plr import (
    BLOCK_KEY,
    apply_paired_block_criterion,
    extract_trial_amplitudes,
    pair_with_flux,
    summarise_block_amplitudes,
)


def _trial(pid, block, cond, test, trial_no, stim_values, baseline=5.0):
    """Build one trial: 30 baseline samples at *baseline*, then *stim_values*."""
    n_base = 30
    times = np.concatenate(
        [
            np.linspace(-1.0, -1.0 / n_base, n_base),
            np.linspace(0.0, 5.0, len(stim_values)),
        ]
    )
    pupil = np.concatenate([np.full(n_base, baseline), np.asarray(stim_values, float)])
    return pd.DataFrame(
        {
            "Participant id": pid,
            "Block": block,
            "Trial type": cond,
            "Test": test,
            "Trial no": trial_no,
            "Trial time Sec": times,
            "Trial phase": ["pre-stim"] * n_base + ["stim"] * len(stim_values),
            "Eye": "R",
            "Right - Size Mm": pupil,
            "Left - Size Mm": np.nan,
        }
    )


def test_maximum_is_taken_per_trial_before_averaging():
    """The block amplitude is the mean of per-trial maxima, not the maximum of
    the mean trace.

    Two trials whose troughs fall at *different* times.  Per-trial maxima are
    both 2.0 mm, so the registered endpoint is 2.0.  Averaging the traces first
    cancels the offset troughs and yields only 1.0, so the two orderings are
    cleanly separated here.
    """
    trial_a = _trial(1, 0, "mel", "a", 1, [5.0, 3.0, 5.0, 5.0, 5.0])
    trial_b = _trial(1, 0, "mel", "a", 2, [5.0, 5.0, 5.0, 3.0, 5.0])
    df = pd.concat([trial_a, trial_b], ignore_index=True)

    trials = extract_trial_amplitudes(df)
    assert set(trials["constriction_amplitude_mm"]) == {2.0}

    blocks = summarise_block_amplitudes(trials)
    amplitude = blocks["mean_constriction_mm"].iloc[0]

    assert amplitude == pytest.approx(2.0), (
        "Block amplitude must be the mean of per-trial maxima (2.0). "
        "Averaging the traces first would give 1.0."
    )


def test_baseline_is_computed_separately_per_trial():
    """A trial's baseline comes from its own pre-stimulus window.

    The two trials sit at different absolute pupil sizes but have identical
    constrictions.  A pooled baseline would make the amplitudes differ.
    """
    trial_a = _trial(1, 0, "mel", "a", 1, [5.0, 4.0, 5.0], baseline=5.0)
    trial_b = _trial(1, 0, "mel", "a", 2, [7.0, 6.0, 7.0], baseline=7.0)
    df = pd.concat([trial_a, trial_b], ignore_index=True)

    trials = extract_trial_amplitudes(df)

    assert trials["constriction_amplitude_mm"].tolist() == pytest.approx([1.0, 1.0])
    assert sorted(trials["baseline_mm"]) == pytest.approx([5.0, 7.0])


def test_trials_are_not_pooled_across_key_fields():
    """Blocks, conditions, tests and trial numbers each keep trials separate.

    Six trials differing only in one key field apiece must yield six trial rows
    and four distinct blocks, not a single pooled series.
    """
    frames = [
        _trial(1, 0, "mel", "a", 1, [5.0, 4.0, 5.0]),
        _trial(1, 0, "mel", "a", 2, [5.0, 4.0, 5.0]),  # different trial no
        _trial(1, 0, "mel", "b", 1, [5.0, 4.0, 5.0]),  # different test
        _trial(1, 1, "mel", "a", 1, [5.0, 4.0, 5.0]),  # different block
        _trial(1, 0, "s", "a", 1, [5.0, 4.0, 5.0]),    # different condition
        _trial(2, 0, "mel", "a", 1, [5.0, 4.0, 5.0]),  # different participant
    ]
    trials = extract_trial_amplitudes(pd.concat(frames, ignore_index=True))
    assert len(trials) == 6
    assert not trials.duplicated(subset=BLOCK_KEY + ["Test", "Trial no"]).any()

    blocks = summarise_block_amplitudes(trials)
    assert len(blocks) == 4, "Expected 4 distinct participant-condition-blocks"

    # The one block holding three trials must report all three.
    pid1_b0_mel = blocks[
        (blocks["Participant id"] == 1)
        & (blocks["Block"] == 0)
        & (blocks["Trial type"] == "mel")
    ]
    assert pid1_b0_mel["n_trials"].iloc[0] == 3
    assert pid1_b0_mel["n_tests"].iloc[0] == 2


def _block_row(pid, block, cond, amp, n_trials=5):
    return {
        "Participant id": pid,
        "Block": block,
        "Trial type": cond,
        "n_trials": n_trials,
        "n_tests": 2,
        "mean_constriction_mm": amp,
        "sd_constriction_mm": 0.1,
        "min_trial_constriction_mm": amp - 0.1,
        "max_trial_constriction_mm": amp + 0.1,
        "exclusion_reason": pd.NA,
        "block_valid": True,
    }


def test_six_block_rule_is_applied_after_pairing():
    """A participant with enough condition blocks but too few *paired* blocks
    is dropped.

    Participant 1 has 7 mel blocks but FLUX in only 4 of them, so it has 4
    paired blocks and must not survive.  Participant 2 has 7 of each and must.
    """
    rows = []
    for b in range(7):
        rows.append(_block_row(1, b, "mel", 0.8))
        if b < 4:
            rows.append(_block_row(1, b, "flux", 1.0))
    for b in range(7):
        rows.append(_block_row(2, b, "mel", 0.8))
        rows.append(_block_row(2, b, "flux", 1.0))
    blocks = pd.DataFrame(rows)

    paired = pair_with_flux(blocks)

    # Pre-pairing, participant 1 would clear a naive six-block count.
    assert (paired["Participant id"] == 1).sum() == 7
    assert paired[paired["Participant id"] == 1]["pair_valid"].sum() == 4

    analysis = apply_paired_block_criterion(paired, min_blocks=6)

    assert 1 not in set(analysis["Participant id"]), (
        "Participant 1 has only 4 paired blocks and must be excluded. "
        "Filtering before pairing would have kept it."
    )
    assert set(analysis["Participant id"]) == {2}
    assert analysis["plr_ratio"].iloc[0] == pytest.approx(0.8)


def test_flux_is_never_in_the_modelled_set():
    """FLUX/FLUX is 1 by construction and must not be returned for modelling."""
    rows = []
    for b in range(7):
        rows.append(_block_row(1, b, "mel", 0.8))
        rows.append(_block_row(1, b, "flux", 1.0))
    paired = pair_with_flux(pd.DataFrame(rows))
    assert "flux" not in set(paired["Trial type"])


def test_duplicate_flux_denominator_is_rejected():
    """Two FLUX rows for one participant-block must fail the join, not silently
    duplicate the numerator rows."""
    rows = [
        _block_row(1, 0, "mel", 0.8),
        _block_row(1, 0, "flux", 1.0),
        _block_row(1, 0, "flux", 1.2),
    ]
    with pytest.raises(Exception):
        pair_with_flux(pd.DataFrame(rows))

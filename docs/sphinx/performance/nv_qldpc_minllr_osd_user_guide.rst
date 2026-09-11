.. The data and figures on this page are reproducible with the scripts in
   benchmarks/qec/nv_qldpc_minllr_osd/ (see the README there for the circuits,
   parameter table, stopping rule and expected runtime).
   Tested at:
     cudaqx  04719bc3dd98540b34976da33f8959061d4f9445
     stim    v1.16.0

.. _minllr_osd_user_guide:

Improving BP+OSD Decoding With Min-LLR OSD Initialization
==========================================================

Belief propagation followed by ordered-statistics decoding (BP+OSD) is the standard
post-processor for quantum LDPC codes: when BP fails to converge, OSD sorts the
columns of the parity-check matrix by how likely BP believes each error is, solves
the syndrome on the most likely columns, and optionally searches low-weight
alternatives (OSD-CS). The quality of the OSD solution therefore depends on the
quality of that column ordering.

By default the ordering is taken from the marginal log-likelihood ratios (LLRs)
after the *last* BP iteration. On circuit-level detector error models, BP that has
not converged is frequently oscillating: a column that BP was confident about at
iteration 6 may look unremarkable at iteration 60. The NV-qLDPC decoder's
``osd_init_method="min_llr"`` option instead orders columns by the *minimum*
marginal LLR each column reached over all BP iterations, so a column that was
confident of an error at *any* point in the trajectory sorts first. The running
minimum is tracked inside the BP kernel, so it costs no extra memory traffic and
needs no LLR history buffer.

.. code-block:: python

    import cudaq_qec as qec

    # From a circuit-level DEM of the code:
    #   H            detector-error matrix (detectors x error mechanisms)
    #   L            observables-flips matrix (observables x error mechanisms)
    #   error_rates  prior probability of each error mechanism
    decoder = qec.get_decoder(
        "nv-qldpc-decoder", H, error_rate_vec=error_rates,
        O=L,                                  # decode straight to observables
        use_sparsity=True, use_osd=True,
        osd_method=3, osd_order=10,           # OSD combination sweep, lambda = 10
        max_iterations=10,                    # only 10 BP iterations ...
        osd_init_method="min_llr",            # ... ordered by the running-min LLR
        bp_batch_size=2048)

    # With O given, each result is the predicted observable flips (k bits),
    # not a correction vector; compare it directly to the measured observables.
    results = decoder.decode_batch(syndromes)
    predicted = [r.result for r in results]

Note the iteration count: ten BP iterations, not the 50-100 that BP+OSD configurations
usually run. With the min-LLR ordering the BP stage only has to *visit* the right
columns at some point, not settle on them, and a short trajectory does that. Below we
compare this configuration (BP10-minLLR+OSD-CS10) against the conventional BP60+OSD-CS10
on bivariate-bicycle (BB) codes as a function of the physical error rate.

Why the ordering matters so much: correlated decoding and Y errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A circuit-level DEM that annotates both the X- and Z-type detectors ("joint XYZ", or
*correlated* decoding) carries strictly more information than one that keeps only the
detectors of the basis being measured ("split X/Z", or *uncorrelated* decoding): a
Y error on a data qubit is a single, correctly-weighted mechanism in the joint DEM but
is modeled as two independent X and Z errors in the split DEMs. In practice, however,
BP+OSD has been observed to decode the *uncorrelated* problem more accurately. The
Tesseract paper [Aghababaie Beni, Higgott, Shutty, `arXiv:2503.10988
<https://arxiv.org/abs/2503.10988>`_] benchmarks against uncorrelated BP+OSD for
exactly this reason, noting that "the uncorrelated variant of BPOSD receives much less
information about the error model" yet is "significantly more accurate than correlated
BPOSD", and explains the effect: "Y-type errors can cause trapping sets in BP-based
decoders when both bases of detectors are annotated" -- every overlapping pair of X and
Z stabilizers produces a 4-cycle through the Y errors on their shared qubits, and the
joint DEM also has more low-weight degenerate configurations (an X and a Z error on the
same qubit are indistinguishable from, and comparable in probability to, a Y error).
"Both degeneracy and short cycles in the Tanner graph are known to be problematic for
BP-based decoders."

Trapping sets and degeneracy are precisely the situations in which BP oscillates rather
than converges, and in which the *final* marginals are a poor summary of what BP
learned. Ordering OSD by the running-minimum LLR recovers the columns that BP was
confident about at some point in the trajectory, before the oscillation set in. The
experiment below shows that with ``min_llr`` the correlated (joint XYZ) BP+OSD decoder
not only stops losing to the uncorrelated one but beats it, so that -- to our knowledge
for the first time with BP+OSD -- decoding the fully correlated error model is strictly
better than decoding the split model.

Experiment
++++++++++

The circuits are the bivariate-bicycle memory-Z experiments published with the
Relay-BP decoder (`github.com/trmue/relay <https://github.com/trmue/relay>`_, in
``tests/testdata/bicycle_bivariate/``, Apache-2.0): 6 and 12 rounds of syndrome
extraction for ``[[72,12,6]]`` and ``[[144,12,12]]`` respectively, under uniform
circuit-level depolarizing noise with one file per physical error rate. Two kinds of
circuit-level DEM are built from them:

* **joint XYZ** -- the circuits as published, which annotate both the X- and Z-check
  detectors; the DEM has all circuit-level error mechanisms (about 16k for
  ``[[72,12,6]]``, 68k for ``[[144,12,12]]``).
* **split X/Z** -- the same circuits with the X-check detectors removed, so only the
  detectors of the basis being decoded remain (the ``assets/benchmarks`` files in this
  repository; 2.2k and 8.8k mechanisms respectively).

All arms decode the same sampled syndromes. An arm is stopped once it has observed
100 logical failures or reached the shot cap; points with zero observed failures are
shown as their 95% Wilson upper bound.

.. list-table:: Decoder and experiment parameters
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Value
   * - GPU
     - NVIDIA GB200
   * - Codes
     - ``[[72,12,6]]`` (6 rounds), ``[[144,12,12]]`` (12 rounds)
   * - Noise model
     - uniform circuit-level, ``p`` swept from 0.001 to 0.005 (``[[72,12,6]]``) and 0.002 to 0.006 (``[[144,12,12]]``)
   * - Shot cap
     - 200,000 (``[[72,12,6]]`` and split X/Z), 100,000 (``[[144,12,12]]`` joint XYZ);
       the ``[[72,12,6]]`` DEMs were sampled a second time with an independent seed and
       a 1000-failure stopping rule, and the ``[[144,12,12]]`` joint-XYZ min-LLR point at
       ``p = 0.002`` was extended with two further independent 1,000,000-shot samples
   * - Stopping rule
     - 100 logical failures per arm (pooled over the independent samples above)
   * - ``bp_method``
     - 0 (sum-product, default)
   * - ``max_iterations``
     - 60 (baseline) / 10 (min-LLR arms)
   * - ``use_osd``
     - True
   * - ``osd_method`` / ``osd_order``
     - 3 (combination sweep) / 10
   * - ``osd_init_method``
     - ``"final_llr"`` (default) / ``"min_llr"``
   * - ``use_sparsity``
     - True
   * - ``bp_batch_size``
     - 2048
   * - Priors
     - ``error_rate_vec`` from the DEM; observables decoded via ``O=L``

Results
+++++++

For each code, the same memory-Z circuit and the same sampled syndromes are decoded
two ways: dashed, faded curves use the split X/Z DEM (uncorrelated decoding), solid
curves the joint XYZ DEM (correlated decoding). Orange is the conventional
BP60+OSD-CS10, blue is BP10-minLLR+OSD-CS10. Shaded bands are 95% Wilson intervals;
each point is stopped at 100 or more logical failures.

.. image:: ../../../assets/docs/minllr_osd_joint_vs_split.png
   :align: center
   :alt: LER versus physical error rate, joint XYZ vs split X/Z DEM, default vs min-LLR OSD ordering

Three observations:

1. **With the default ordering, correlated decoding loses** (solid orange above dashed
   orange). This reproduces the Tesseract paper's observation: the correlated decoder is
   *worse* than the uncorrelated one by 6x-7x for ``[[72,12,6]]`` at ``p`` <= 0.002 and by
   14x-19x for ``[[144,12,12]]`` at ``p`` = 0.002-0.003, despite having strictly more
   information about the noise.

2. **The min-LLR ordering removes the penalty and inverts it** (solid blue below dashed
   blue). On the joint XYZ DEMs the change from BP60+OSD-CS10 to BP10-minLLR+OSD-CS10
   lowers the logical error rate by 3.6x at ``p = 0.005``, 11x at 0.003, 25x at 0.002 and
   36x at 0.001 (2.0e-3 vs 5.5e-5) for ``[[72,12,6]]``, and by 2.4x at ``p = 0.006``, 5x at
   0.005, 15x at 0.004, 46x at 0.003 (4.5e-2 vs 9.7e-4) and 117x at ``p = 0.002`` (3.5e-3 vs
   3.0e-5, the latter from 63 failures in 2.1 million shots) for ``[[144,12,12]]``. The gap
   widens as ``p`` falls because the min-LLR curves keep the steep low-``p`` slope of about
   5 while the default-ordering curves flatten toward saturation. The correlated min-LLR
   decoder now beats the uncorrelated one on both codes: by 2.5x-3.6x for ``[[72,12,6]]`` at
   ``p`` <= 0.002 (5.5e-5 vs 2.0e-4 at ``p = 0.001``) and 1.2x-1.6x at higher rates, and by
   2.0x at ``p = 0.002`` (3.0e-5 vs 6.0e-5) and 1.3x at ``p = 0.003`` for ``[[144,12,12]]``,
   with ``p`` >= 0.004 within noise. To our knowledge this is the first BP+OSD
   configuration for which decoding the full correlated circuit-level model is at least
   as accurate as decoding the split model at every rate measured, and better wherever
   the two are resolvable.

3. **On the split X/Z DEMs the ordering matters much less** (dashed blue only modestly
   below dashed orange): 1.2x-1.6x for ``[[72,12,6]]`` and 1.1x-3.1x for ``[[144,12,12]]``,
   largest at the lowest rate. Those DEMs have no Y-induced 4-cycles, so BP oscillates
   less and the final marginals are already a reasonable ordering. The min-LLR gain is
   concentrated exactly where the trapping sets are.

On the iteration count: BP converges on only about 1% of joint-XYZ syndromes within 10
iterations at these rates (and on well under half even at 60 for ``p`` >= 0.002), so on
those DEMs the decoder
is effectively "OSD with a good ordering", and ten iterations are enough to produce
that ordering. We have not swept the iteration count systematically here; anecdotally,
comparable results have been seen with as few as five iterations, which suggests the
useful signal sits in the earliest part of the BP trajectory. The savings are real: six
times fewer BP iterations than the baseline, at a lower logical error rate.

See Also
++++++++

* :ref:`Quantum Low-Density Parity-Check Decoder <qldpc_decoder>` -- the nv-qldpc-decoder overview
* :ref:`Improving Relay BP Decoding With Gamma Ensembles <ensemble_gamma_user_guide>`
* :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference

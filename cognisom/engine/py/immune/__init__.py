"""Immune subsystem: closed-loop neoantigen presentation (Upgrade 2).

Modules
-------
mhc_loading
    Score peptides against patient HLA alleles and return presentations.
tcr_repertoire
    Stochastic TCR repertoire with 16-dim feature-based pMHC affinity.
tcell_kill
    Kill probability from affinity x MHC-I level x costimulation.
nk_receptor
    KIR-HLA ligand groups, NK licensing, and the activating/inhibitory
    balance that replaces the one-line missing-self step.
macrophage_phagocytosis
    The PD-1 "don't eat me" gate on phagocytosis (Gordon 2017).
macrophage_polarization
    The M1/M2 axis: its cytokine drivers, the relaxation toward a local
    set point, kill potency and cytokine output as functions of
    phenotype, and the arginase gate M2 macrophages impose on T cells.
sympathetic
    beta2-adrenergic attenuation of T-cell kill function.
escape_variation
    The per-division rates and magnitudes of heritable immune escape:
    loss of one HLA class I ALLELE (partial, still recognised -- not a
    whole haplotype, which would be three alleles in cis) and the
    second hit that takes a cell out of the T-cell loop altogether.
treg_suppression
    TGF-beta repression of the cytolytic machinery by regulatory T
    cells, scaled by the patient's Treg fraction. Also records, with
    the measurement, why MDSCs are NOT a second gate here: their
    mechanism is the arginine depletion the arginase gate already
    models, and that gate is already at its floor.
"""

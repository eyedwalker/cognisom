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
"""

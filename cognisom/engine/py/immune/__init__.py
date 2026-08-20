"""Immune subsystem: closed-loop neoantigen presentation (Upgrade 2).

Modules
-------
mhc_loading
    Score peptides against patient HLA alleles and return presentations.
tcr_repertoire
    Stochastic TCR repertoire with 16-dim feature-based pMHC affinity.
tcell_kill
    Kill probability from affinity x MHC-I level x costimulation.
"""

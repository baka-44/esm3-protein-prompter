"""
proteinredesign.config_builders — deterministic, per-preset config builders.

These translate scientist-level structured inputs (PDB + residue# + AA name +
intent) into the exact config each engine needs (MPNN fixed_positions, RFdiffusion
contig map / atomic motif / partial_T). Per decision B5, **code — never an LLM —
generates the positional config**; an optional Claude layer may only parse free
text into the structured spec fed to these builders.
"""

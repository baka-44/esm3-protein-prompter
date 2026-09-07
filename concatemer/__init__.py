"""
concatemer — assemble and screen peptide-concatemer proteins for secretory expression.

The product is the HYDROLYSATE, not the protein: a carrier chain is expressed and secreted in
Pichia, then cleaved to release a defined blend of short bioactive peptides. So the objective is
not "does it fold" but "does it get made, and does it give back exactly the peptides we designed
in". See `digest.py` for that objective and `spec.py` for the design IR.

Architecture assumption (A, "disordered by design"): the chain is engineered NOT to fold. ER
quality control penalises exposed hydrophobic surface, unpaired cysteines and aggregation — not
disorder as such — so a polar, highly charged, Cys-free chain traverses the pathway with nothing
for BiP to grab. Structure-derived features are therefore deliberately absent from this package.
"""

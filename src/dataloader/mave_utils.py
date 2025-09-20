"""
MAVE filtering utilities for genomic foundation model training.
Combines method categorization, filtering, and basic analysis functions.
"""

MAVE_METHODS = {
    # Canonical DMS modalities (treat these as the highest-quality, most-standardized bucket)
    "JOINT_DMS": [
        "DMS-BarSeq", "DMS-TileSeq", "deep mutational scan", "Deep mutational scan", "complementation",
    ],

    # Cis-regulatory assays (MPRA/promoter/enhancer/ultraconserved elements; timepoints/replicates included)
    "REGULATORY_FUNCTION": [
        "MPRA"
    ],

    # Protein–protein / protein–ligand interaction readouts
    "BINDING_INTERACTION": [
        "yeast two-hybrid", "Y2H", "phage display"
    ],

    # Stability/biophysical tolerance (protease susceptibility, HPLC stability series, folding)
    "BIOPHYSICAL_STABILITY": [
        "trypsin digestion", "chymotrypsin digestion"
    ],

    # RNA-level abundance/processing
    "ABUNDANCE_RNA": [
        "RNA abundance"
    ],

    # Protein abundance & translation efficiency (flow/FACS, fluorescence levels, polysomes)
    "ABUNDANCE_PROTEIN_TRANSLATION": [
        "protein abundance", "flow cytometry", "polysome"
    ],

    # Organismal/cellular growth phenotypes
    "FITNESS_GROWTH": [
        "fitness", "growth", "escape"
    ],

    # Computation-derived or post-processed outputs (use with caution for training targets)
    "COMPUTATIONAL_PROCESSED": [
        "Enrich2", "regression scores", "combined scores"
    ],

    # Annotations to exclude from training (batch effects, controls, processed data)
    "EXCLUDE_FROM_TRAINING": [
        "Individual Synonymous Wildtype Scores", "Individual Synonymous Variant Scores",
        "control", "Control", "scrambled control", "siRNA control",
        "synthetic construct", "artificial"
    ],
}

def get_method_categories():
    """Get available MAVE method categories."""
    return list(MAVE_METHODS.keys())

def expand_method_filters(methods):
    """
    Expand category names to individual methods.

    Args:
        methods: List of method names or categories

    Returns:
        List of individual method names
    """
    expanded = []
    for method in methods:
        if method.upper() in MAVE_METHODS:
            expanded.extend(MAVE_METHODS[method.upper()])
        else:
            expanded.append(method)
    return expanded

# Basic sequence analysis utilities
def calculate_gc_content(sequence):
    """Calculate GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)

def is_low_complexity(sequence, threshold=0.3):
    """Check if sequence has low complexity (simple repeats)."""
    if len(sequence) < 20:
        return False

    # Check for homopolymer runs
    for base in 'ATCG':
        if base * 8 in sequence.upper():  # 8+ consecutive identical bases
            return True

    # Check k-mer diversity
    kmers = set()
    k = 3
    for i in range(len(sequence) - k + 1):
        kmers.add(sequence[i:i+k])

    max_possible = min(4**k, len(sequence) - k + 1)
    complexity = len(kmers) / max_possible if max_possible > 0 else 0

    return complexity < threshold

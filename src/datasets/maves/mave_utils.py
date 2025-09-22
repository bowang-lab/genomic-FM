"""
MAVE filtering utilities for genomic foundation model training.
Combines method categorization, filtering, and basic analysis functions.
"""

from mavehgvs.util import parse_variant_strings

MAVE_METHODS = {
    # Canonical DMS modalities (treat these as the highest-quality, most-standardized bucket)
    "DMS": [
        "DMS-BarSeq", "DMS-TileSeq", "deep mutational scan", "Deep mutational scan", "Deep Mutational Scan",
    ],

    # Cis-regulatory assays (MPRA/promoter/enhancer/ultraconserved elements; timepoints/replicates included)
    "REGULATORY": [
        "MPRA", "saturation mutagenesis"
    ],

    # Stability/biophysical tolerance (protease susceptibility)
    "BIOPHYSICAL_STABILITY": [
        "trypsin digestion", "chymotrypsin digestion"
    ],

    # RNA-level abundance/processing
    "RNA_ABUNDANCE": [
        "RNA abundance",
    ],

    # Protein abundance
    "PROTEIN_ABUNDANCE": [
        "protein abundance",
    ],

    # Protein translational efficiency
    "PROTEIN_TRANSLATION": [
        "polysome",
    ],

    # Organismal/cellular growth phenotypes
    "ESCAPE": [
        "escape",
    ],

    # Computation-derived or post-processed outputs (use with caution for training targets)
    "COMPUTATIONAL_PROCESSED": [
        "Enrich2", "regression scores", "combined scores"
    ],

    # Annotations to exclude from training (batch effects, controls, processed data)
    "EXCLUDE_FROM_TRAINING": [
        "control",
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

def get_variant_position_annotations(hgvs_variant):
    """
    Extract position annotations from a HGVS variant using VariantPosition methods.

    Args:
        hgvs_variant: HGVS variant string

    Returns:
        dict: Dictionary containing position annotations including:
            - is_utr: True if position is in UTR
            - is_intronic: True if position is intronic
            - intronic_position: Number of bases into intron (if intronic)
            - is_extended: True if uses extended position syntax
            - position_type: String description of position type
    """
    try:
        variants, errors = parse_variant_strings([hgvs_variant])

        if errors[0] is not None or variants[0] is None:
            return {
                'is_utr': None,
                'is_intronic': None,
                'intronic_position': None,
                'is_extended': None,
                'position_type': 'unknown'
            }

        variant = variants[0]
        positions = variant.positions

        # Handle different position types
        if isinstance(positions, tuple):
            # Range variant - use first position for annotation
            pos_obj = positions[0]
        elif isinstance(positions, list):
            # Multi-variant - use first position for annotation
            pos_obj = positions[0] if positions else None
        else:
            # Single position
            pos_obj = positions

        if pos_obj is None:
            return {
                'is_utr': None,
                'is_intronic': None,
                'intronic_position': None,
                'is_extended': None,
                'position_type': 'unknown'
            }

        # Extract annotations using VariantPosition methods
        is_utr = pos_obj.is_utr() if hasattr(pos_obj, 'is_utr') else None
        is_intronic = pos_obj.is_intronic() if hasattr(pos_obj, 'is_intronic') else None
        intronic_position = pos_obj.intronic_position if hasattr(pos_obj, 'intronic_position') else None
        is_extended = pos_obj.is_extended() if hasattr(pos_obj, 'is_extended') else None

        # Determine position type description
        if is_utr:
            position_type = 'UTR'
        elif is_intronic:
            position_type = 'intronic'
        elif is_extended:
            position_type = 'extended'
        else:
            # Determine from HGVS prefix
            if hgvs_variant.startswith('c.'):
                position_type = 'coding'
            elif hgvs_variant.startswith('n.'):
                position_type = 'noncoding'
            elif hgvs_variant.startswith('g.'):
                position_type = 'genomic'
            else:
                position_type = 'standard'

        return {
            'is_utr': is_utr,
            'is_intronic': is_intronic,
            'intronic_position': intronic_position,
            'is_extended': is_extended,
            'position_type': position_type
        }

    except Exception as e:
        return {
            'is_utr': None,
            'is_intronic': None,
            'intronic_position': None,
            'is_extended': None,
            'position_type': 'error'
        }

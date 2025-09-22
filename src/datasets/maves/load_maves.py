import re
import requests
from io import StringIO

import pandas as pd
from mavehgvs.exceptions import MaveHgvsParseError
from mavehgvs.patterns.dna import (
    dna_del_c, dna_del_gmo, dna_del_n,
    dna_delins_c, dna_delins_gmo, dna_delins_n,
    dna_dup_c, dna_dup_gmo, dna_dup_n,
    dna_equal_c, dna_equal_gmo, dna_equal_n,
    dna_ins_c, dna_ins_gmo, dna_ins_n,
    dna_multi_variant, dna_nt, dna_single_variant,
    dna_sub_c, dna_sub_gmo, dna_sub_n,
    dna_variant_c, dna_variant_gmo, dna_variant_n
)
from mavehgvs.patterns.util import remove_named_groups, combine_patterns
from mavehgvs.position import VariantPosition
from mavehgvs.util import parse_variant_strings
from tqdm import tqdm

from .mave_utils import get_variant_position_annotations

def get_all_urn_ids():
    """Fetch all URN IDs from MAVE-DB experiments API."""
    response = requests.get("https://api.mavedb.org/api/v1/experiments/")
    if response.status_code == 200:
        return [item['urn'] for item in response.json() if item['urn'].startswith("urn:")]
    print(f"Error: {response.status_code}")
    return []

def extract_target_info(json_data):
    """Extract target gene information from JSON data."""
    results = []
    for item in json_data:
        target_info = {
            'title': item.get('title', ''),
            'description': item.get('shortDescription', ''),
            'urn': item.get('urn', ''),
            'numVariants': item.get('numVariants', ''),
            'targetGenes': []
        }
        for gene in item.get('targetGenes', []):
            ids = {i.get('identifier', {}).get('dbName'): i.get('identifier', {}).get('identifier')
                   for i in gene.get('externalIdentifiers', [])}
            target_info['targetGenes'].append({
                'gene_name': gene.get('name', ''),
                'sequence': gene.get('targetSequence', {}).get('sequence', ''),
                'sequence_type': gene.get('targetSequence', {}).get('sequenceType', ''),
                'reference_genome': gene.get('targetSequence', {}).get('reference', {}).get('shortName', ''),
                'Ensembl': ids.get('Ensembl'),
                'UniProt': ids.get('UniProt')
            })
        results.append(target_info)
    return results

def get_scores(urn_id):
    response = requests.get(f"https://api.mavedb.org/api/v1/score-sets/{urn_id}/scores")
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    print(f"Error: {response.status_code}")
    return pd.DataFrame()

def get_score_set(urn_id):
    response = requests.get(f"https://api.mavedb.org/api/v1/experiments/{urn_id}/score-sets")
    if response.status_code == 200:
        return extract_target_info(response.json())
    print(f"Error: {response.status_code}")
    return []

# Pre-compile combined patterns for better performance
_coding_pattern = None
_noncoding_pattern = None
_genomic_pattern = None

def _get_combined_patterns():
    """
    Create combined patterns using mavehgvs utilities for optimal performance.
    """
    global _coding_pattern, _noncoding_pattern, _genomic_pattern

    if _coding_pattern is None:
        # Combine coding patterns
        coding_patterns = [
            remove_named_groups(dna_sub_c),
            remove_named_groups(dna_del_c),
            remove_named_groups(dna_ins_c),
            remove_named_groups(dna_dup_c),
            remove_named_groups(dna_delins_c),
            remove_named_groups(dna_equal_c),
            remove_named_groups(dna_variant_c)
        ]
        _coding_pattern = combine_patterns(coding_patterns)

    if _noncoding_pattern is None:
        # Combine non-coding patterns
        noncoding_patterns = [
            remove_named_groups(dna_sub_n),
            remove_named_groups(dna_del_n),
            remove_named_groups(dna_ins_n),
            remove_named_groups(dna_dup_n),
            remove_named_groups(dna_delins_n),
            remove_named_groups(dna_equal_n),
            remove_named_groups(dna_variant_n)
        ]
        _noncoding_pattern = combine_patterns(noncoding_patterns)

    if _genomic_pattern is None:
        # Combine genomic patterns
        genomic_patterns = [
            remove_named_groups(dna_sub_gmo),
            remove_named_groups(dna_del_gmo),
            remove_named_groups(dna_ins_gmo),
            remove_named_groups(dna_dup_gmo),
            remove_named_groups(dna_delins_gmo),
            remove_named_groups(dna_equal_gmo),
            remove_named_groups(dna_variant_gmo)
        ]
        _genomic_pattern = combine_patterns(genomic_patterns)

    return _coding_pattern, _noncoding_pattern, _genomic_pattern

def validate_hgvs_with_patterns(hgvs_nt):
    """
    Optimized HGVS validation using combined patterns for better performance.
    """
    try:
        # First check if it's a multi-variant (has brackets)
        if '[' in hgvs_nt and ']' in hgvs_nt:
            return bool(re.fullmatch(dna_multi_variant, hgvs_nt))

        # For single variants, test the most comprehensive pattern first
        if re.fullmatch(dna_single_variant, hgvs_nt):
            return True

        # Get pre-compiled combined patterns for fallback
        coding_pattern, noncoding_pattern, genomic_pattern = _get_combined_patterns()

        # Use specific combined patterns as fallback
        if hgvs_nt.startswith('c.'):
            return bool(re.fullmatch(coding_pattern, hgvs_nt))
        elif hgvs_nt.startswith('n.'):
            return bool(re.fullmatch(noncoding_pattern, hgvs_nt))
        elif hgvs_nt.startswith('g.'):
            return bool(re.fullmatch(genomic_pattern, hgvs_nt))
        else:
            # For unknown prefixes, try original individual patterns
            patterns_to_test = [
                dna_sub_c, dna_del_c, dna_ins_c, dna_dup_c,
                dna_delins_c, dna_equal_c, dna_variant_c,
                dna_sub_n, dna_del_n, dna_ins_n, dna_dup_n,
                dna_delins_n, dna_equal_n, dna_variant_n,
                dna_sub_gmo, dna_del_gmo, dna_ins_gmo, dna_dup_gmo,
                dna_delins_gmo, dna_equal_gmo, dna_variant_gmo
            ]
            for pattern in patterns_to_test:
                if re.fullmatch(pattern, hgvs_nt):
                    return True
            return False

    except Exception:
        return False

def extract_variant_positions(hgvs_variant):
    """
    Enhanced position extraction using VariantPosition for better validation.

    Returns:
        tuple: (max_position, diagnostic_info)
    """
    try:
        # Parse just to get position information without requiring target sequence
        variants, errors = parse_variant_strings([hgvs_variant])

        if errors[0] is not None:
            return None, f"Parse error: {errors[0]}"

        variant = variants[0]
        if variant is None:
            return None, "Failed to parse variant"

        # Extract position information using the API
        positions = variant.positions

        if isinstance(positions, tuple):
            # Range variant - validate each position is a VariantPosition
            valid_positions = []
            for pos in positions:
                if isinstance(pos, VariantPosition) and pos.position is not None:
                    valid_positions.append(pos.position)

            if not valid_positions:
                return None, "No valid positions found in range variant"

            max_pos = max(valid_positions)
            min_pos = min(valid_positions)
            return max_pos, f"Range variant: {min_pos}-{max_pos}"
        elif isinstance(positions, list):
            # Multi-variant case - handle list of positions
            valid_positions = []
            for pos in positions:
                if isinstance(pos, VariantPosition) and pos.position is not None:
                    valid_positions.append(pos.position)
                elif hasattr(pos, 'position') and pos.position is not None:
                    valid_positions.append(pos.position)

            if not valid_positions:
                return None, "No valid positions found in multi-variant"

            max_pos = max(valid_positions)
            min_pos = min(valid_positions)
            return max_pos, f"Multi-variant: {min_pos}-{max_pos}"
        else:
            # Single position variant - validate it's a VariantPosition
            if isinstance(positions, VariantPosition) and positions.position is not None:
                return positions.position, f"Single position: {positions.position}"
            elif hasattr(positions, 'position') and positions.position is not None:
                return positions.position, f"Single position: {positions.position}"
            else:
                return None, f"Invalid position object: {type(positions)}"

    except Exception as e:
        return None, f"Exception during position extraction: {e}"


def get_alternate_dna_sequence(dna_sequence, hgvs_nt, verbose=True):
    """
    Apply MAVE-HGVS variant to DNA sequence.

    Handles legacy _wt format and skips invalid X variants.
    Provides better diagnostics for coordinate issues.

    Args:
        dna_sequence: Target DNA sequence
        hgvs_nt: HGVS variant string
        verbose: If True, print warnings for skipped variants
    """
    # Handle legacy _wt format (Enrich2 syntax -> MAVE-HGVS c.=)
    if hgvs_nt == '_wt':
        return dna_sequence

    # Handle empty or None input
    if not hgvs_nt or hgvs_nt.strip() == '':
        return dna_sequence

    # Skip variants with X (ambiguity characters not supported by MAVE-HGVS)
    if 'X>' in hgvs_nt or '>X' in hgvs_nt:
        if verbose:
            print(f"Warning: Skipping variant with ambiguity character X: {hgvs_nt}")
        return None

    # Validate HGVS pattern using comprehensive pattern matching
    if not validate_hgvs_with_patterns(hgvs_nt):
        if verbose:
            print(f"Warning: Invalid HGVS pattern format: {hgvs_nt}")
        return None

    # Use MAVE-HGVS parser with proper error handling
    try:
        variants, errors = parse_variant_strings([hgvs_nt], targetseq=dna_sequence)

        if errors[0] is not None:
            error_msg = str(errors[0])

            # For coordinate errors, provide better diagnostics
            if any(phrase in error_msg for phrase in [
                "coordinate out of bounds", "out of bounds", "Invalid position"
            ]):
                # Extract position information for better error reporting
                max_pos, pos_info = extract_variant_positions(hgvs_nt)
                if max_pos and verbose:
                    print(f"Warning: Skipping incompatible variant {hgvs_nt}: {pos_info} exceeds sequence length {len(dna_sequence)}")
                elif verbose:
                    print(f"Warning: Skipping incompatible variant {hgvs_nt}: {error_msg}")
                return None
            elif "reference does not match target" in error_msg:
                if verbose:
                    print(f"Warning: Skipping incompatible variant {hgvs_nt}: {error_msg}")
                return None
            else:
                if verbose:
                    print(f"Error: Could not parse variant {hgvs_nt}: {error_msg}")
                return None

        variant = variants[0]
        if variant is None:
            if verbose:
                print(f"Warning: Could not parse variant {hgvs_nt}")
            return None

        # Apply the variant to get alternate sequence
        return apply_variant_to_sequence(dna_sequence, variant)

    except MaveHgvsParseError as e:
        if verbose:
            print(f"Warning: MAVE-HGVS parse error for {hgvs_nt}: {str(e)}")
        return None
    except Exception as e:
        if verbose:
            print(f"Warning: Unexpected error parsing variant {hgvs_nt}: {str(e)}")
        return None

def apply_variant_to_sequence(dna_sequence, variant):
    """
    Apply a MAVE-HGVS variant to a DNA sequence to get the alternate sequence.
    Handles both single variants and multi-variants using the official library.
    """
    # Handle equality variants (wild-type)
    if variant.is_target_identical():
        return dna_sequence

    # Handle multi-variants by processing individual variant tuples
    if variant.is_multi_variant():
        return apply_multi_variant_to_sequence(dna_sequence, variant)

    # Handle single variants
    return apply_single_variant_to_sequence(dna_sequence, variant)

def apply_multi_variant_to_sequence(dna_sequence, multi_variant):
    """
    Apply a multi-variant by processing each individual variant in the multi-variant.
    According to the MAVE-HGVS documentation, multi-variants need to be split
    and each individual variant re-parsed using single-variant patterns.
    """
    # Get the string representation of the multi-variant
    multi_variant_str = str(multi_variant)

    # Extract the prefix and variant part
    if '.' in multi_variant_str:
        prefix, variants_part = multi_variant_str.split('.', 1)
    else:
        print(f"Warning: Invalid multi-variant format: {multi_variant_str}")
        return dna_sequence

    # Remove brackets from the variant part
    if variants_part.startswith('[') and variants_part.endswith(']'):
        variants_part = variants_part[1:-1]

    # Split by semicolon to get individual variant strings
    individual_variant_strings = [v.strip() for v in variants_part.split(';') if v.strip()]

    # Apply each individual variant sequentially
    current_sequence = dna_sequence
    for variant_str in individual_variant_strings:
        # Reconstruct full HGVS string for this individual variant
        full_variant_str = f"{prefix}.{variant_str}"

        # Parse the individual variant using the MAVE-HGVS parser
        variants, errors = parse_variant_strings([full_variant_str], targetseq=current_sequence)

        if errors[0] is not None:
            print(f"Warning: Could not parse individual variant {full_variant_str}: {errors[0]}")
            continue

        if variants[0] is None:
            print(f"Warning: Could not parse individual variant {full_variant_str}")
            continue

        individual_variant = variants[0]

        # Apply this individual variant to the current sequence
        new_sequence = apply_single_variant_to_sequence(current_sequence, individual_variant)
        if new_sequence is None:
            print(f"Warning: Failed to apply individual variant {full_variant_str}")
            return dna_sequence  # Return original sequence on failure

        current_sequence = new_sequence

    return current_sequence

def apply_single_variant_to_sequence(dna_sequence, variant):
    """
    Apply a single MAVE-HGVS variant to a DNA sequence.
    """
    # Get variant type - should be a string according to API
    vtype = variant.variant_type

    # Get position from VariantPosition object
    pos_obj = variant.positions
    if isinstance(pos_obj, tuple) and len(pos_obj) > 0:
        # Multiple positions (ranges) - get first position
        pos = pos_obj[0].position
    else:
        # Single VariantPosition object
        pos = pos_obj.position

    if vtype == 'sub':
        # Single substitution - access sequence tuple directly
        ref_seq, alt_seq = variant.sequence
        pos_idx = pos - 1  # Convert to 0-based indexing
        return dna_sequence[:pos_idx] + alt_seq + dna_sequence[pos_idx + 1:]

    elif vtype == 'del':
        # Handle deletions - may have start/end positions
        positions = variant.positions
        if isinstance(positions, tuple) and len(positions) == 2:
            # Range deletion
            start_pos = positions[0].position - 1  # 0-based
            end_pos = positions[1].position  # Exclusive end
            return dna_sequence[:start_pos] + dna_sequence[end_pos:]
        else:
            # Single position deletion
            pos_idx = pos - 1
            return dna_sequence[:pos_idx] + dna_sequence[pos_idx + 1:]

    elif vtype == 'ins':
        # Insertion between start and end positions
        positions = variant.positions
        start_pos = positions[0].position  # Insert after this position
        inserted_seq = variant.sequence
        return dna_sequence[:start_pos] + inserted_seq + dna_sequence[start_pos:]

    elif vtype == 'dup':
        # Duplication - may have start/end positions
        positions = variant.positions
        if isinstance(positions, tuple) and len(positions) == 2:
            # Range duplication
            start_pos = positions[0].position - 1  # 0-based
            end_pos = positions[1].position  # Exclusive end
            dup_seq = dna_sequence[start_pos:end_pos]
            return dna_sequence[:end_pos] + dup_seq + dna_sequence[end_pos:]
        else:
            # Single position duplication
            pos_idx = pos - 1
            dup_base = dna_sequence[pos_idx]
            return dna_sequence[:pos] + dup_base + dna_sequence[pos:]

    elif vtype == 'delins':
        # Deletion-insertion
        replacement_seq = variant.sequence
        positions = variant.positions
        if isinstance(positions, tuple) and len(positions) == 2:
            # Range delins
            start_pos = positions[0].position - 1  # 0-based
            end_pos = positions[1].position  # Exclusive end
            return dna_sequence[:start_pos] + replacement_seq + dna_sequence[end_pos:]
        else:
            # Single position delins
            pos_idx = pos - 1
            return dna_sequence[:pos_idx] + replacement_seq + dna_sequence[pos_idx + 1:]

    else:
        print(f"Warning: Unsupported variant type: {vtype}")
        return dna_sequence



def get_maves(Seq_length=1024, limit = None, target='score', sequence_type='dna', region_type=None, verbose_warnings=True):
    urn_ids = get_all_urn_ids()
    avail = 0
    total = 0
    n_studies = 0
    data = []

    for study_num, urn_id in tqdm(enumerate(urn_ids), total=len(urn_ids), desc="Processing studies"):
        score_set = get_score_set(urn_id)

        if limit and study_num >= limit:
            break

        for index, exp in enumerate(score_set):
            if not exp.get('targetGenes'):  # Check if targetGenes is empty or not present
                print(f"Warning: No target genes found for {urn_id}")
                continue  # Skip this entry if no target genes

            urn_id = exp.get('urn', None)
            title = exp.get('title', None)
            description = exp.get('description', None)
            numVariants = exp.get('numVariants', None)
            exp_sequence_type = exp.get('targetGenes', None)[0]['sequence_type']
            annotation = ': '.join([title, description])

            total += int(numVariants)
            scores = get_scores(urn_id)
            if isinstance(scores, pd.DataFrame) and (exp_sequence_type == sequence_type):
                if not scores.empty:
                    if index == 0:
                        n_studies += 1

                    # Get sequence once for this experiment
                    ref_sequence = exp["targetGenes"][0]["sequence"]
                    sequence_length = len(ref_sequence)

                    # Check if sequence fits within the requested length
                    if sequence_length <= Seq_length:
                        for index, row in scores.iterrows():
                            if pd.notna(row['hgvs_nt']) and pd.notna(row["score"]):
                                # Filter based on sequence_type
                                hgvs_prefix = row['hgvs_nt'].split('.')[0]
                                if region_type == 'coding' and hgvs_prefix != 'c':
                                    continue
                                elif region_type == 'noncoding' and hgvs_prefix != 'n':
                                    continue

                                # Skip X variants early (before any processing)
                                if 'X>' in row['hgvs_nt'] or '>X' in row['hgvs_nt']:
                                    if verbose_warnings:
                                        print(f"Warning: Skipping variant with ambiguity character X: {row['hgvs_nt']}")
                                    continue

                                # Single validation and processing in get_alternate_dna_sequence
                                alt = get_alternate_dna_sequence(ref_sequence, row['hgvs_nt'], verbose=verbose_warnings)
                                if alt is not None:
                                    avail += 1

                                    # Get position annotations
                                    position_annotations = get_variant_position_annotations(row['hgvs_nt'])

                                    # Build annotation string with position information
                                    position_info = f"Position Type: {position_annotations['position_type']}"
                                    if position_annotations['is_utr']:
                                        position_info += ", UTR"
                                    if position_annotations['is_intronic']:
                                        position_info += f", Intronic (pos: {position_annotations['intronic_position']})"
                                    if position_annotations['is_extended']:
                                        position_info += ", Extended Syntax"

                                    x = [ref_sequence, alt, f"{annotation}, Sequence Type: {exp_sequence_type}, HGVS Prefix: {hgvs_prefix}, {position_info}, URN: {urn_id}, HGVS: {row['hgvs_nt']}"]
                                    y = row[target]
                                    data.append([x,y])
                    else:
                        # Sequence too long - skip this entire experiment
                        if verbose_warnings:
                            print(f"Warning: Skipping experiment {urn_id}: sequence length {sequence_length} exceeds limit {Seq_length}")
                else:
                    print(f"Error: Scores dataframe is empty: {urn_id}")
            else:
                print(f"Error: Could not retrieve scores for: {urn_id}")
    print(f"Total number of studies: {n_studies}/{len(urn_ids)}")
    print(f"Total number of MAVEs: {avail}/{total}")
    return data
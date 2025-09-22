"""MAVE-DB data loader with comprehensive HGVS variant processing using mavehgvs."""

import logging
import re
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import requests
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
from mavehgvs.patterns.util import combine_patterns
from mavehgvs.position import VariantPosition
from mavehgvs.util import parse_variant_strings
from tqdm import tqdm

from .mave_utils import get_variant_position_annotations

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
MAVE_DB_BASE_URL = "https://api.mavedb.org/api/v1"
REQUEST_TIMEOUT = None  # No timeout
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# Comprehensive pattern compilation using hierarchical mavehgvs DNA patterns
HGVS_VALIDATION_PATTERN = re.compile(combine_patterns([
    dna_single_variant, dna_multi_variant
], groupname='hgvs_variants'))
NT_PATTERN = re.compile(dna_nt)

# Custom comprehensive patterns using combine_patterns for correct precedence
# Order matters: delins before del, since delins contains 'del' substring
VARIANT_C_PATTERN = re.compile(combine_patterns([
    dna_equal_c, dna_sub_c, dna_delins_c, dna_del_c, dna_ins_c, dna_dup_c
], groupname='variant_c'))

VARIANT_N_PATTERN = re.compile(combine_patterns([
    dna_equal_n, dna_sub_n, dna_delins_n, dna_del_n, dna_ins_n, dna_dup_n
], groupname='variant_n'))

VARIANT_GMO_PATTERN = re.compile(combine_patterns([
    dna_equal_gmo, dna_sub_gmo, dna_delins_gmo, dna_del_gmo, dna_ins_gmo, dna_dup_gmo
], groupname='variant_gmo'))

# Combined equality pattern for filtering identity variants
EQUALITY_PATTERN = re.compile(combine_patterns([
    dna_equal_c, dna_equal_n, dna_equal_gmo
], groupname='equality'))

# Create session for connection reuse
_api_session = None

def _get_api_session() -> requests.Session:
    """Get or create a reusable requests session."""
    global _api_session
    if _api_session is None:
        _api_session = requests.Session()
        _api_session.headers.update({'User-Agent': 'MAVE-Loader/1.0'})
    return _api_session


def _make_api_request(url: str, retries: int = RETRY_ATTEMPTS) -> Optional[requests.Response]:
    """Make API request with retry logic and proper error handling.

    Args:
        url: The URL to request
        retries: Number of retry attempts

    Returns:
        Response object if successful, None otherwise
    """
    session = _get_api_session()

    for attempt in range(retries):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                wait_time = min(10, RETRY_DELAY + attempt)  # Cap at 10 seconds
                logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{retries}")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"API request failed with status {response.status_code}: {url}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request exception on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)

    logger.error(f"Failed to fetch data after {retries} attempts: {url}")
    return None

def get_all_urn_ids() -> List[str]:
    """Fetch all URN IDs from MAVE-DB experiments API.

    Returns:
        List of URN IDs starting with 'urn:', empty list if fetch fails
    """
    url = f"{MAVE_DB_BASE_URL}/experiments/"
    response = _make_api_request(url)

    if response is None:
        logger.error("Failed to fetch URN IDs")
        return []

    try:
        data = response.json()
        return [item['urn'] for item in data if item.get('urn', '').startswith("urn:")]
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing URN IDs response: {e}")
        return []

def extract_target_info(json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract target gene information from JSON data.

    Args:
        json_data: List of experiment dictionaries from MAVE-DB API

    Returns:
        List of processed target information dictionaries
    """
    results = []

    for item in json_data:
        try:
            target_info = {
                'title': item.get('title', ''),
                'description': item.get('shortDescription', ''),
                'urn': item.get('urn', ''),
                'numVariants': item.get('numVariants', 0),
                'targetGenes': []
            }

            for gene in item.get('targetGenes', []):
                try:
                    # Safely extract external identifiers
                    ids = {}
                    for ext_id in gene.get('externalIdentifiers', []):
                        if isinstance(ext_id, dict) and 'identifier' in ext_id:
                            id_info = ext_id['identifier']
                            if isinstance(id_info, dict):
                                db_name = id_info.get('dbName')
                                identifier = id_info.get('identifier')
                                if db_name and identifier:
                                    ids[db_name] = identifier

                    # Extract target sequence information
                    target_seq = gene.get('targetSequence', {})
                    reference = target_seq.get('reference', {})

                    gene_info = {
                        'gene_name': gene.get('name', ''),
                        'sequence': target_seq.get('sequence', ''),
                        'sequence_type': target_seq.get('sequenceType', ''),
                        'reference_genome': reference.get('shortName', '') if isinstance(reference, dict) else '',
                        'Ensembl': ids.get('Ensembl'),
                        'UniProt': ids.get('UniProt')
                    }
                    target_info['targetGenes'].append(gene_info)

                except Exception as e:
                    logger.warning(f"Error processing gene info for {item.get('urn', 'unknown')}: {e}")
                    continue

            results.append(target_info)

        except Exception as e:
            logger.warning(f"Error processing target info for {item.get('urn', 'unknown')}: {e}")
            continue

    return results

def get_scores(urn_id: str) -> pd.DataFrame:
    """Fetch variant scores for a given URN ID.

    Args:
        urn_id: The URN identifier for the score set

    Returns:
        DataFrame containing variant scores, empty DataFrame if fetch fails
    """
    url = f"{MAVE_DB_BASE_URL}/score-sets/{urn_id}/scores"
    response = _make_api_request(url)

    if response is None:
        logger.warning(f"Failed to fetch scores for {urn_id}")
        return pd.DataFrame()

    try:
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        logger.warning(f"Error parsing scores CSV for {urn_id}: {e}")
        return pd.DataFrame()

def get_score_set(urn_id: str) -> List[Dict[str, Any]]:
    """Fetch score set information for a given URN ID.

    Args:
        urn_id: The URN identifier for the experiment

    Returns:
        List of target information dictionaries, empty list if fetch fails
    """
    url = f"{MAVE_DB_BASE_URL}/experiments/{urn_id}/score-sets"
    response = _make_api_request(url)

    if response is None:
        logger.warning(f"Failed to fetch score set for {urn_id}")
        return []

    try:
        data = response.json()
        return extract_target_info(data)
    except (ValueError, KeyError) as e:
        logger.warning(f"Error parsing score set response for {urn_id}: {e}")
        return []


def _validate_hgvs_format(hgvs_nt: str) -> bool:
    """Validate HGVS string using comprehensive mavehgvs patterns.

    Args:
        hgvs_nt: HGVS variant string to validate

    Returns:
        True if string matches valid HGVS format, False otherwise
    """
    return HGVS_VALIDATION_PATTERN.match(hgvs_nt) is not None

def _validate_position(position) -> None:
    """Validate position using VariantPosition methods.

    Args:
        position: Position object to validate

    Raises:
        MaveHgvsParseError: If position is intronic (not supported)
    """
    if isinstance(position, VariantPosition) and position.is_intronic():
        raise MaveHgvsParseError(f"Intronic positions not supported: {position}")

def _validate_nucleotide_sequence(seq: str, context: str) -> None:
    """Validate nucleotide sequence using dna_nt pattern.

    Args:
        seq: Nucleotide sequence to validate
        context: Context description for error messages

    Raises:
        MaveHgvsParseError: If sequence contains invalid nucleotides
    """
    if seq and not NT_PATTERN.match(seq):
        raise MaveHgvsParseError(f"Invalid nucleotide sequence '{seq}' in {context}")

def _parse_variant_details(variant: str, prefix: str = None) -> Optional[Dict[str, Any]]:
    """Parse variant details using appropriate comprehensive mavehgvs patterns.

    Args:
        variant: Single variant string (without prefix)
        prefix: HGVS prefix (c, n, g, m, o) to determine correct pattern

    Returns:
        Dictionary with variant details or None if no match
    """
    # Select appropriate pattern based on prefix
    if prefix == 'c':
        pattern = VARIANT_C_PATTERN
        pattern_type = 'coding'
    elif prefix == 'n':
        pattern = VARIANT_N_PATTERN
        pattern_type = 'non-coding'
    elif prefix in ['g', 'm', 'o']:
        pattern = VARIANT_GMO_PATTERN
        pattern_type = 'genomic'
    else:
        # Try all patterns if prefix unknown
        patterns_to_try = [
            (VARIANT_C_PATTERN, 'coding'),
            (VARIANT_N_PATTERN, 'non-coding'),
            (VARIANT_GMO_PATTERN, 'genomic')
        ]
        for pattern, pattern_type in patterns_to_try:
            match = pattern.match(variant)
            if match:
                break
        else:
            return None

    if prefix:
        match = pattern.match(variant)
        if not match:
            return None

    groups = match.groupdict()
    result = {
        'prefix': prefix,
        'pattern_type': pattern_type,
        'groups': groups
    }

    # Determine specific variant type from active groups
    # With combine_patterns, we get renamed groups like variant_c_dna_delins_c
    variant_type = 'unknown'
    for key, value in groups.items():
        if value and 'dna_delins_' in key:
            variant_type = 'delins'
            break
        elif value and 'dna_sub_' in key:
            variant_type = 'substitution'
            break
        elif value and 'dna_del_' in key:
            variant_type = 'deletion'
            break
        elif value and 'dna_ins_' in key:
            variant_type = 'insertion'
            break
        elif value and 'dna_dup_' in key:
            variant_type = 'duplication'
            break
        elif value and 'dna_equal_' in key:
            variant_type = 'equality'
            break

    result['type'] = variant_type

    # Extract position information (look for any position-related key)
    position = None
    for key, value in groups.items():
        if value and 'position' in key:
            position = int(value)
            break
        elif value and 'start' in key:
            start_pos = int(value)
            end_key = key.replace('start', 'end')
            end_pos = int(groups.get(end_key, start_pos))
            position = max(start_pos, end_pos)
            result['start'] = start_pos
            result['end'] = end_pos
            break

    if position:
        result['position'] = position

    # Extract reference base for substitutions
    for key, value in groups.items():
        if value and 'ref' in key:
            result['ref_base'] = value.upper()
            break

    # Extract alternate sequence
    for key, value in groups.items():
        if value and ('new' in key or 'seq' in key):
            result['sequence'] = value.upper()
            break

    return result

def _preprocess_multi_variant_hgvs(hgvs_nt: str, sequence_length: int = None, reference_sequence: str = None) -> str:
    """Remove target-identical variants (=) and validate variants from multi-variant HGVS strings.

    Args:
        hgvs_nt: The HGVS variant string
        sequence_length: Optional length of the target sequence for bounds checking
        reference_sequence: Optional reference sequence for validation

    Returns:
        Preprocessed HGVS string with invalid variants removed
    """
    # Check if this is a bracketed variant string (multi-variant or single variant in brackets)
    if not hgvs_nt.startswith(('c.[', 'n.[', 'g.[', 'o.[')):
        return hgvs_nt

    # Extract the prefix and the variant list
    prefix = hgvs_nt.split('.')[0] + '.'
    variant_part = hgvs_nt[len(prefix):]

    if not (variant_part.startswith('[') and variant_part.endswith(']')):
        return hgvs_nt

    # Extract variants between brackets
    variants_str = variant_part[1:-1]  # Remove brackets
    # Handle both multi-variant (with semicolons) and single variant (without semicolons)
    if ';' in variants_str:
        variants = [v.strip() for v in variants_str.split(';')]
    else:
        variants = [variants_str.strip()]

    # Filter out ALL target-identical variants using combined equality pattern
    # The mavehgvs library does not allow ANY identity variants in multi-variant strings
    non_identical_variants = [
        variant for variant in variants
        if not EQUALITY_PATTERN.match(variant.strip())
    ]

    # Filter variants based on position bounds and reference validation
    if sequence_length is not None or reference_sequence is not None:
        valid_variants = []
        # Extract prefix for consistent parsing
        hgvs_prefix = prefix.rstrip('.')

        for variant in non_identical_variants:
            # Parse variant details using appropriate pattern for this prefix
            details = _parse_variant_details(variant, hgvs_prefix)

            if details:
                position = details.get('position')
                ref_base = details.get('ref_base')
                is_valid = True

                # Validate position bounds
                if position and sequence_length and position > sequence_length:
                    is_valid = False

                # Validate reference base for substitutions
                if (is_valid and ref_base and reference_sequence and
                    position and position <= len(reference_sequence)):
                    actual_base = reference_sequence[position - 1].upper()
                    if actual_base != ref_base:
                        is_valid = False

                if is_valid:
                    valid_variants.append(variant)

        non_identical_variants = valid_variants

    # If all variants were filtered out, return a simple synonymous variant
    if not non_identical_variants:
        return prefix + '1='  # Simple synonymous variant

    # If only one variant remains, return it as a simple variant
    if len(non_identical_variants) == 1:
        return prefix + non_identical_variants[0]

    # Otherwise, reconstruct the multi-variant string
    return prefix + '[' + ';'.join(non_identical_variants) + ']'

def get_alternate_dna_sequence(dna_sequence: str, hgvs_nt: str, verbose: bool = True) -> Optional[str]:
    """Apply HGVS variant to DNA sequence using mavehgvs.

    Args:
        dna_sequence: Reference DNA sequence
        hgvs_nt: HGVS variant string (e.g., 'c.123A>T', 'n.[1A>C;2T>G]')
        verbose: Whether to log warnings for invalid variants

    Returns:
        Modified DNA sequence if variant is valid, None if invalid or fails
    """
    if hgvs_nt in ('_wt', '') or not hgvs_nt.strip() or 'X>' in hgvs_nt or '>X' in hgvs_nt:
        return dna_sequence if hgvs_nt in ('_wt', '') or not hgvs_nt.strip() else None

    # Handle synonymous variants using combined equality pattern
    # These indicate no change to the sequence
    if EQUALITY_PATTERN.match(hgvs_nt.strip()):
        return dna_sequence

    # Early validation for single variants using specific pattern parsing
    if not hgvs_nt.startswith(('c.[', 'n.[', 'g.[', 'o.[')):
        # Extract prefix and variant part for parsing
        if '.' in hgvs_nt:
            prefix, variant_part = hgvs_nt.split('.', 1)
        else:
            prefix, variant_part = None, hgvs_nt

        details = _parse_variant_details(variant_part, prefix)

        if details:
            position = details.get('position')
            ref_base = details.get('ref_base')

            # Validate position bounds
            if position and position > len(dna_sequence):
                if verbose:
                    logging.warning(f"Position {position} out of bounds for sequence length {len(dna_sequence)} in variant '{hgvs_nt}'")
                return None

            # Validate reference base for substitutions
            if ref_base and position and position <= len(dna_sequence):
                actual_base = dna_sequence[position - 1].upper()
                if actual_base != ref_base:
                    if verbose:
                        logging.warning(f"Failed to parse HGVS variant '{hgvs_nt}': variant reference does not match target")
                    return None

    # Preprocess multi-variant HGVS to remove target-identical variants and validate positions/references
    processed_hgvs = _preprocess_multi_variant_hgvs(hgvs_nt, len(dna_sequence), dna_sequence)

    # If preprocessing resulted in a simple synonymous variant, return original sequence
    if EQUALITY_PATTERN.match(processed_hgvs.strip()):
        return dna_sequence

    if not _validate_hgvs_format(processed_hgvs):
        raise MaveHgvsParseError(f"Invalid HGVS format: {processed_hgvs}")

    # Extract expected prefix for validation
    expected_prefix = processed_hgvs.split('.')[0] if '.' in processed_hgvs else None

    # Use mavehgvs with target sequence validation and prefix checking
    variants, errors = parse_variant_strings([processed_hgvs],
                                           targetseq=dna_sequence,
                                           expected_prefix=expected_prefix)

    if errors[0] or not variants[0]:
        if verbose:
            logging.warning(f"Failed to parse HGVS variant '{hgvs_nt}': {errors[0] if errors[0] else 'No variant parsed'}")
        return None

    variant = variants[0]

    # Use Variant class methods for better handling
    if variant.is_target_identical():
        return dna_sequence

    # For multi-variants, log additional info
    if variant.is_multi_variant() and verbose:
        logging.debug(f"Processing multi-variant HGVS: {hgvs_nt}")

    current_sequence = list(dna_sequence)
    for variant_type, position, sequence_data in variant.variant_tuples():
        if position is None:
            if verbose:
                logging.debug(f"Skipping variant with None position in HGVS '{hgvs_nt}': variant_type={variant_type}, sequence_data={sequence_data}")
            continue
        _validate_position(position)

        if isinstance(position, tuple):
            # For tuple positions, use the first element as the starting position
            start_pos = position[0]
            if start_pos is None:
                continue  # Skip variants with None position
            pos_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
        elif isinstance(position, VariantPosition):
            pos_idx = position.position - 1
        elif position is not None:
            pos_idx = position - 1
        else:
            continue  # Skip variants with None position

        if variant_type == 'sub':
            ref, alt = sequence_data
            _validate_nucleotide_sequence(ref, "substitution reference")
            _validate_nucleotide_sequence(alt, "substitution alternative")
            if pos_idx < 0 or pos_idx >= len(current_sequence):
                # Use debug level for out-of-bounds positions as they're now filtered earlier
                if verbose:
                    logging.debug(f"Position index {pos_idx} out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                continue
            current_sequence[pos_idx] = alt

        elif variant_type == 'del':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                if start_idx < 0 or start_idx >= len(current_sequence) or end_idx < 0 or end_idx > len(current_sequence):
                    if verbose:
                        logging.debug(f"Deletion range [{start_idx}:{end_idx}] out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                del current_sequence[start_idx:end_idx]
            else:
                if pos_idx < 0 or pos_idx >= len(current_sequence):
                    if verbose:
                        logging.debug(f"Position index {pos_idx} out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                del current_sequence[pos_idx]

        elif variant_type == 'ins':
            _validate_nucleotide_sequence(sequence_data, "insertion")
            if pos_idx < -1 or pos_idx >= len(current_sequence):
                if verbose:
                    logging.debug(f"Insertion position {pos_idx} out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                continue
            for i, base in enumerate(sequence_data):
                current_sequence.insert(pos_idx + 1 + i, base)

        elif variant_type == 'dup':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                if start_idx < 0 or start_idx >= len(current_sequence) or end_idx < 0 or end_idx > len(current_sequence):
                    if verbose:
                        logging.debug(f"Duplication range [{start_idx}:{end_idx}] out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                dup_seq = current_sequence[start_idx:end_idx]
                current_sequence[end_idx:end_idx] = dup_seq
            else:
                if pos_idx < 0 or pos_idx >= len(current_sequence):
                    if verbose:
                        logging.debug(f"Position index {pos_idx} out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                current_sequence.insert(pos_idx + 1, current_sequence[pos_idx])

        elif variant_type == 'delins':
            # For delins, sequence_data is just the alternate sequence string
            alt = sequence_data
            _validate_nucleotide_sequence(alt, "deletion-insertion")
            if isinstance(position, tuple):
                start_pos, end_pos = position
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                if start_idx < 0 or start_idx >= len(current_sequence) or end_idx < 0 or end_idx > len(current_sequence):
                    if verbose:
                        logging.debug(f"Delins range [{start_idx}:{end_idx}] out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                current_sequence[start_idx:end_idx] = list(alt)
            else:
                if pos_idx < 0 or pos_idx >= len(current_sequence):
                    if verbose:
                        logging.debug(f"Position index {pos_idx} out of bounds for sequence length {len(current_sequence)} in variant '{hgvs_nt}'")
                    continue
                current_sequence[pos_idx] = alt

    return ''.join(current_sequence)


def get_maves(seq_length=1024, limit=None, target='score', sequence_type='dna', region_type=None, verbose_warnings=True):
    """Load MAVE data with comprehensive HGVS processing.

    Args:
        seq_length: Maximum sequence length to include (default: 1024)
        limit: Maximum number of studies to process (default: None)
        target: Target column name to extract (default: 'score')
        sequence_type: Type of sequence to filter for (default: 'dna')
        region_type: Region type filter ('coding', 'noncoding', or None)
        verbose_warnings: Whether to show detailed warnings (default: True)

    Returns:
        List of [[ref_sequence, alt_sequence, metadata], score] pairs
    """
    urn_ids = get_all_urn_ids()
    avail = total = n_studies = 0
    data = []

    for study_num, urn_id in tqdm(enumerate(urn_ids), total=len(urn_ids), desc="Processing studies"):
        if limit and study_num >= limit:
            break

        score_set = get_score_set(urn_id)
        for index, exp in enumerate(score_set):
            if not exp.get('targetGenes'):
                continue

            exp_urn_id = exp.get('urn')
            exp_sequence_type = exp['targetGenes'][0]['sequence_type']

            if exp_sequence_type != sequence_type:
                continue

            total += exp.get('numVariants', 0)
            scores = get_scores(exp_urn_id)

            if not isinstance(scores, pd.DataFrame) or scores.empty:
                continue

            if index == 0:
                n_studies += 1

            ref_sequence = exp['targetGenes'][0]['sequence']
            if len(ref_sequence) > seq_length:
                continue

            annotation = f"{exp.get('title', '')}: {exp.get('description', '')}"

            for _, row in scores.iterrows():
                if pd.isna(row['hgvs_nt']) or pd.isna(row['score']):
                    continue

                hgvs_prefix = row['hgvs_nt'].split('.')[0]
                if (region_type == 'coding' and hgvs_prefix != 'c') or \
                   (region_type == 'noncoding' and hgvs_prefix != 'n'):
                    continue

                alt = get_alternate_dna_sequence(ref_sequence, row['hgvs_nt'], verbose=verbose_warnings)
                if alt is not None:
                    avail += 1
                    position_annotations = get_variant_position_annotations(row['hgvs_nt'])

                    position_info = f"Position Type: {position_annotations['position_type']}"
                    for flag, label in [('is_utr', 'UTR'), ('is_extended', 'Extended Syntax')]:
                        if position_annotations[flag]:
                            position_info += f", {label}"
                    if position_annotations['is_intronic']:
                        position_info += f", Intronic (pos: {position_annotations['intronic_position']})"

                    metadata = f"{annotation}, Sequence Type: {exp_sequence_type}, HGVS Prefix: {hgvs_prefix}, {position_info}, URN: {exp_urn_id}, HGVS: {row['hgvs_nt']}"
                    data.append([[ref_sequence, alt, metadata], row[target]])

    logger.info(f"Total studies: {n_studies}/{len(urn_ids)}, MAVEs: {avail}/{total}")
    return data
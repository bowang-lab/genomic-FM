"""MAVE-DB data loader with comprehensive HGVS variant processing using mavehgvs."""

import logging
import re
import time
from functools import lru_cache
from io import StringIO
from typing import Dict, List, Optional, Tuple, Any

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
RETRY_DELAY = 2  # seconds

# Keep only necessary patterns for nucleotide validation
NT_PATTERN = re.compile(dna_nt)

@lru_cache(maxsize=1)
def _get_api_session() -> requests.Session:
    """Get or create a reusable requests session."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'MAVE-Loader/1.0'})
    return session


def _make_api_request(url: str) -> Optional[requests.Response]:
    """Make API request with infinite retry logic.

    Args:
        url: The URL to request

    Returns:
        Response object if successful, None for permanent failures
    """
    session = _get_api_session()

    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 502, 503, 504]:  # Rate limited or server errors
                error_type = "Rate limited" if response.status_code == 429 else f"Server error ({response.status_code})"
                logger.warning(f"{error_type} on attempt {attempt}, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                logger.error(f"API request failed with permanent error {response.status_code}: {url}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request exception on attempt {attempt}: {e}, retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

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
        return pd.read_csv(StringIO(response.text), low_memory=False)
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

def _get_position_index(position, current_sequence_length: int) -> Optional[int]:
    """Extract position index from various position types.

    Args:
        position: Position object (VariantPosition, int, tuple, etc.)
        current_sequence_length: Length of current sequence for bounds checking

    Returns:
        Zero-based position index or None if invalid
    """
    if position is None:
        return None

    if isinstance(position, tuple):
        start_pos = position[0]
        if start_pos is None:
            return None
        pos_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
    elif isinstance(position, VariantPosition):
        pos_idx = position.position - 1
    else:
        pos_idx = position - 1

    return pos_idx if 0 <= pos_idx < current_sequence_length else None

def _is_position_valid(pos_idx: Optional[int], sequence_length: int, allow_insertion: bool = False) -> bool:
    """Check if position index is valid for sequence operations.

    Args:
        pos_idx: Position index to validate
        sequence_length: Length of the sequence
        allow_insertion: Whether to allow insertion position (-1)

    Returns:
        True if position is valid
    """
    if pos_idx is None:
        return False
    min_pos = -1 if allow_insertion else 0
    return min_pos <= pos_idx < sequence_length

def _has_intronic_positions(hgvs_variant: str) -> bool:
    """Check if HGVS variant contains intronic positions.

    Args:
        hgvs_variant: Complete HGVS variant string

    Returns:
        True if variant contains intronic positions
    """
    variants, errors = parse_variant_strings([hgvs_variant])
    if errors[0] or not variants[0]:
        return False

    for variant_type, position, sequence_data in variants[0].variant_tuples():
        if position is None:
            continue
        if isinstance(position, tuple):
            if any(isinstance(p, VariantPosition) and p.is_intronic() for p in position):
                return True
        elif isinstance(position, VariantPosition) and position.is_intronic():
            return True
    return False

def _filter_variants(hgvs_variant: str, reference_sequence: str = None) -> Optional[str]:
    """Filter out intronic variants and synonymous variants using mavehgvs.

    Args:
        hgvs_variant: HGVS variant string
        reference_sequence: Optional reference sequence for validation

    Returns:
        Filtered HGVS variant or None if should be skipped
    """
    # Skip intronic variants
    if _has_intronic_positions(hgvs_variant):
        return None

    # Parse with mavehgvs for validation
    variants, errors = parse_variant_strings([hgvs_variant], targetseq=reference_sequence)
    if errors[0] or not variants[0]:
        return None

    # Skip synonymous variants
    if variants[0].is_target_identical():
        return None

    return hgvs_variant

def get_alternate_dna_sequence(dna_sequence: str, hgvs_nt: str, verbose: bool = True) -> Optional[str]:
    """Apply HGVS variant to DNA sequence using mavehgvs.

    Args:
        dna_sequence: Reference DNA sequence
        hgvs_nt: HGVS variant string (e.g., 'c.123A>T', 'n.[1A>C;2T>G]')
        verbose: Whether to log warnings for invalid variants

    Returns:
        Modified DNA sequence if variant is valid, None if invalid or fails
    """
    # Handle empty/wildtype cases
    if hgvs_nt in ('_wt', '') or not hgvs_nt.strip():
        return dna_sequence

    # Skip invalid variants
    if 'X>' in hgvs_nt or '>X' in hgvs_nt:
        return None

    # Filter out intronic variants and validate
    filtered_variant = _filter_variants(hgvs_nt, dna_sequence)
    if filtered_variant is None:
        if verbose and _has_intronic_positions(hgvs_nt):
            logging.warning(f"Intronic variant '{hgvs_nt}' not supported for coding sequences")
        return None

    # Parse and apply variant using mavehgvs
    expected_prefix = hgvs_nt.split('.')[0] if '.' in hgvs_nt else None
    variants, errors = parse_variant_strings([hgvs_nt], targetseq=dna_sequence, expected_prefix=expected_prefix)

    if errors[0] or not variants[0]:
        if verbose:
            logging.warning(f"Failed to parse HGVS variant '{hgvs_nt}': {errors[0] or 'No variant parsed'}")
        return None

    variant = variants[0]

    # Return original sequence for synonymous variants
    if variant.is_target_identical():
        return dna_sequence

    # Apply variant modifications using mavehgvs variant_tuples
    current_sequence = list(dna_sequence)
    for variant_type, position, sequence_data in variant.variant_tuples():
        if position is None:
            continue

        _validate_position(position)
        pos_idx = _get_position_index(position, len(current_sequence))

        if variant_type == 'sub':
            ref, alt = sequence_data
            _validate_nucleotide_sequence(ref, "substitution reference")
            _validate_nucleotide_sequence(alt, "substitution alternative")
            if _is_position_valid(pos_idx, len(current_sequence)):
                current_sequence[pos_idx] = alt

        elif variant_type == 'del':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = (start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1)
                end_idx = (end_pos.position if isinstance(end_pos, VariantPosition) else end_pos)
                if 0 <= start_idx < len(current_sequence) and 0 < end_idx <= len(current_sequence):
                    del current_sequence[start_idx:end_idx]
            elif _is_position_valid(pos_idx, len(current_sequence)):
                del current_sequence[pos_idx]

        elif variant_type == 'ins':
            _validate_nucleotide_sequence(sequence_data, "insertion")
            if _is_position_valid(pos_idx, len(current_sequence), allow_insertion=True):
                for i, base in enumerate(sequence_data):
                    current_sequence.insert(pos_idx + 1 + i, base)

        elif variant_type == 'dup':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = (start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1)
                end_idx = (end_pos.position if isinstance(end_pos, VariantPosition) else end_pos)
                if 0 <= start_idx < len(current_sequence) and 0 < end_idx <= len(current_sequence):
                    dup_seq = current_sequence[start_idx:end_idx]
                    current_sequence[end_idx:end_idx] = dup_seq
            elif _is_position_valid(pos_idx, len(current_sequence)):
                current_sequence.insert(pos_idx + 1, current_sequence[pos_idx])

        elif variant_type == 'delins':
            _validate_nucleotide_sequence(sequence_data, "deletion-insertion")
            if isinstance(position, tuple):
                start_pos, end_pos = position
                start_idx = (start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1)
                end_idx = (end_pos.position if isinstance(end_pos, VariantPosition) else end_pos)
                if 0 <= start_idx < len(current_sequence) and 0 < end_idx <= len(current_sequence):
                    current_sequence[start_idx:end_idx] = list(sequence_data)
            elif _is_position_valid(pos_idx, len(current_sequence)):
                current_sequence[pos_idx] = sequence_data

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
    processed_count = total_count = study_count = 0
    data = []

    for study_idx, urn_id in tqdm(enumerate(urn_ids), total=len(urn_ids), desc="Processing studies"):
        if limit and study_idx >= limit:
            break

        score_set = get_score_set(urn_id)
        for exp_idx, exp in enumerate(score_set):
            target_genes = exp.get('targetGenes')
            if not target_genes or target_genes[0]['sequence_type'] != sequence_type:
                continue

            ref_sequence = target_genes[0]['sequence']
            if len(ref_sequence) > seq_length:
                continue

            total_count += exp.get('numVariants', 0)
            scores = get_scores(exp.get('urn'))

            if not isinstance(scores, pd.DataFrame) or scores.empty:
                continue

            if exp_idx == 0:
                study_count += 1

            title = exp.get('title', '')
            description = exp.get('description', '')
            annotation = f"{title}: {description}"

            for _, row in scores.iterrows():
                if pd.isna(row['hgvs_nt']) or pd.isna(row['score']):
                    continue

                hgvs_prefix = row['hgvs_nt'].split('.')[0]
                if ((region_type == 'coding' and hgvs_prefix != 'c') or
                    (region_type == 'noncoding' and hgvs_prefix != 'n')):
                    continue

                alt_sequence = get_alternate_dna_sequence(ref_sequence, row['hgvs_nt'], verbose=verbose_warnings)
                if alt_sequence is None:
                    continue

                processed_count += 1
                pos_annotations = get_variant_position_annotations(row['hgvs_nt'])

                position_info = f"Position Type: {pos_annotations['position_type']}"
                if pos_annotations['is_utr']:
                    position_info += ", UTR"
                if pos_annotations['is_extended']:
                    position_info += ", Extended Syntax"
                if pos_annotations['is_intronic']:
                    position_info += f", Intronic (pos: {pos_annotations['intronic_position']})"

                metadata = (f"{annotation}, Sequence Type: {target_genes[0]['sequence_type']}, "
                           f"HGVS Prefix: {hgvs_prefix}, {position_info}, "
                           f"URN: {exp.get('urn')}, HGVS: {row['hgvs_nt']}")

                data.append([[ref_sequence, alt_sequence, metadata], row[target]])

    logger.info(f"Total studies: {study_count}/{len(urn_ids)}, MAVEs: {processed_count}/{total_count}")
    return data
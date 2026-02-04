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
REQUEST_TIMEOUT = 30  # seconds
RETRY_DELAY = 2  # seconds
MAX_RETRIES = 10

# Pattern Compilation for HGVS Validation
# Using combine_patterns for efficient matching
NT_PATTERN = re.compile(dna_nt)

# Combined comprehensive validation pattern
ALL_DNA_VARIANTS = re.compile(
    combine_patterns([dna_single_variant, dna_multi_variant], groupname="any_variant")
)

# Positioned equal pattern using existing mavehgvs patterns
# dna_equal_c is most comprehensive (handles intronic positions, extended syntax)
# dna_equal_n handles bare '=' for non-coding sequences
# dna_equal_gmo handles genomic positions
POSITIONED_EQUAL_PATTERN = re.compile(
    combine_patterns([dna_equal_c, dna_equal_n, dna_equal_gmo], groupname="positioned_equal")
)

# Prefix-specific patterns for targeted validation
DNA_VARIANT_PATTERNS = {
    'c': re.compile(dna_variant_c),
    'n': re.compile(dna_variant_n),
    'g': re.compile(dna_variant_gmo),
    'm': re.compile(dna_variant_gmo),
    'o': re.compile(dna_variant_gmo)
}

@lru_cache(maxsize=1)
def _get_api_session() -> requests.Session:
    """Get or create a reusable requests session."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'MAVE-Loader/1.0'})
    return session


def _make_api_request(url: str) -> Optional[requests.Response]:
    """Make API request with retry logic.

    Args:
        url: The URL to request

    Returns:
        Response object if successful, None for permanent failures or after MAX_RETRIES
    """
    session = _get_api_session()

    # Retryable status codes
    RETRYABLE_CODES = {429, 502, 503, 504}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response
            elif response.status_code in RETRYABLE_CODES:
                error_type = "Rate limited" if response.status_code == 429 else f"Server error ({response.status_code})"
                logger.warning(f"{error_type} on attempt {attempt}/{MAX_RETRIES}, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                logger.error(f"API request failed with permanent error {response.status_code}: {url}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request exception on attempt {attempt}/{MAX_RETRIES}: {e}, retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    logger.error(f"API request failed after {MAX_RETRIES} attempts: {url}")
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

def _extract_external_identifiers(gene: Dict[str, Any]) -> Dict[str, str]:
    """Extract external identifiers from gene data.

    Args:
        gene: Gene dictionary from API response

    Returns:
        Dictionary mapping database names to identifiers
    """
    ids = {}
    for ext_id in gene.get('externalIdentifiers', []):
        if not isinstance(ext_id, dict) or 'identifier' not in ext_id:
            continue
        id_info = ext_id['identifier']
        if isinstance(id_info, dict):
            db_name = id_info.get('dbName')
            identifier = id_info.get('identifier')
            if db_name and identifier:
                ids[db_name] = identifier
    return ids

def extract_target_info(json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract target gene information from JSON data.

    Args:
        json_data: List of experiment dictionaries from MAVE-DB API

    Returns:
        List of processed target information dictionaries
    """
    results = []

    for item in json_data:
        urn = item.get('urn', 'unknown')
        try:
            target_info = {
                'title': item.get('title', ''),
                'description': item.get('shortDescription', ''),
                'urn': urn,
                'numVariants': item.get('numVariants', 0),
                'targetGenes': []
            }

            for gene in item.get('targetGenes', []):
                try:
                    ids = _extract_external_identifiers(gene)
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
                    logger.warning(f"Error processing gene info for {urn}: {e}")
                    continue

            results.append(target_info)

        except Exception as e:
            logger.warning(f"Error processing target info for {urn}: {e}")
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





def _extract_position_index(position, sequence_type: str = None) -> Optional[int]:
    """Extract 0-based position index from various position types.

    Args:
        position: Position object (VariantPosition, int, tuple)
        sequence_type: Type of sequence ('c', 'n', 'g', etc.) for context

    Returns:
        Zero-based position index or None
    """
    if position is None:
        return None

    if isinstance(position, tuple):
        # For ranges, use the start position
        pos = position[0]
    else:
        pos = position

    if isinstance(pos, VariantPosition):
        base_pos = pos.position
        if base_pos is None:
            return None

        # Handle intronic positions
        if pos.is_intronic():
            intronic_offset = pos.intronic_position or 0

            # Intronic positions are only valid for certain sequence types
            if sequence_type == 'n':
                # Non-coding sequences can include intronic regions
                if intronic_offset > 0:
                    # Position is intronic_offset bases after the base position
                    return (base_pos - 1) + intronic_offset
                elif intronic_offset < 0:
                    # Negative offset requires knowledge of intron length
                    logger.debug(f"Cannot map negative intronic offset {intronic_offset} without intron context")
                    return None
            elif sequence_type == 'c':
                # For coding sequences, intronic positions refer to splice sites
                # These don't map to positions in the cDNA reference
                logger.debug(f"Intronic position {pos} in coding sequence - indicates splice variant")
                return None
            else:
                # Genomic sequences (g, m, o) shouldn't have intronic positions
                # as they are contiguous sequences including introns
                logger.warning(f"Unexpected intronic position {pos} for sequence type '{sequence_type}'")
                return None

        # Non-intronic position
        return base_pos - 1

    elif isinstance(pos, int):
        return pos - 1

    return None


def _validate_variant_format(hgvs_variant: str) -> bool:
    """Validate HGVS variant format using efficient pattern matching.

    Args:
        hgvs_variant: HGVS variant string to validate

    Returns:
        True if format is valid
    """
    # Handle gene-prefixed variants (e.g., RAF1:n.1141G>A)
    variant_to_check = hgvs_variant
    if ':' in hgvs_variant:
        # Split on colon to separate gene name from HGVS variant
        _, variant_to_check = hgvs_variant.split(':', 1)

    # Quick check with combined pattern
    if ALL_DNA_VARIANTS.match(variant_to_check):
        return True

    # Check for positioned synonymous variants (e.g., n.270=, c.123=)
    if '.' in variant_to_check:
        prefix, suffix = variant_to_check.split('.', 1)
        if prefix in ['c', 'n', 'g', 'm', 'o'] and POSITIONED_EQUAL_PATTERN.match(suffix):
            return True

    # Fallback to prefix-specific validation
    if '.' in variant_to_check:
        prefix = variant_to_check.split('.', 1)[0]
        pattern = DNA_VARIANT_PATTERNS.get(prefix)
        if pattern:
            return bool(pattern.match(variant_to_check))

    return False

def _parse_and_validate_variant(hgvs_variant: str, reference_sequence: str = None, verbose: bool = False) -> Optional[Any]:
    """Parse and validate HGVS variant using mavehgvs with pattern pre-validation.

    Args:
        hgvs_variant: HGVS variant string
        reference_sequence: Optional reference sequence for validation
        verbose: Whether to log detailed information

    Returns:
        Parsed variant object or None if invalid
    """
    hgvs_variant = hgvs_variant.strip()

    # Extract the actual HGVS variant part (handle gene-prefixed variants)
    variant_part = hgvs_variant
    if ':' in hgvs_variant:
        _, variant_part = hgvs_variant.split(':', 1)

    # Handle positioned synonymous variants (e.g., n.270=, c.123=)
    if '.' in variant_part:
        prefix, suffix = variant_part.split('.', 1)
        if prefix in ['c', 'n', 'g', 'm', 'o'] and POSITIONED_EQUAL_PATTERN.match(suffix):
            return None  # These are synonymous/wildtype

    # Handle multi-variants with synonymous components by filtering them out
    if re.match(r'^[cnhgmo]\.\[.*=.*\]$', variant_part):
        match = re.match(r'^([cnhgmo])\.\[(.+)\]$', variant_part)
        if match:
            prefix = match.group(1)
            components = match.group(2).split(';')
            # Filter out synonymous components (those ending with =)
            non_synonymous = [comp.strip() for comp in components if not comp.strip().endswith('=')]

            if not non_synonymous:
                return None  # All components are synonymous

            # Reconstruct variant with only non-synonymous components
            if len(non_synonymous) == 1:
                new_variant_part = f'{prefix}.{non_synonymous[0]}'
            else:
                new_variant_part = f'{prefix}.[{";".join(non_synonymous)}]'

            # Reconstruct full variant with gene prefix if it existed
            if ':' in hgvs_variant:
                gene_part, _ = hgvs_variant.split(':', 1)
                hgvs_variant = f'{gene_part}:{new_variant_part}'
            else:
                hgvs_variant = new_variant_part

    # Pre-validate format using patterns
    if not _validate_variant_format(hgvs_variant):
        return None

    # Determine expected prefix safely from the actual sequence prefix
    expected_prefix = None
    variant_for_prefix = hgvs_variant
    if ':' in hgvs_variant:
        _, variant_for_prefix = hgvs_variant.split(':', 1)

    if '.' in variant_for_prefix and not variant_for_prefix.startswith('.'):
        prefix = variant_for_prefix.split('.')[0]
        # Only set expected_prefix for valid prefixes
        if prefix in ['c', 'n', 'g', 'm', 'o', 'p', 'r']:
            expected_prefix = prefix

    variants, errors = parse_variant_strings([hgvs_variant], targetseq=reference_sequence, expected_prefix=expected_prefix)

    if errors[0] or not variants[0]:
        return None

    variant = variants[0]

    # Log intronic positions for debugging
    if verbose:
        for variant_type, position, _ in variant.variant_tuples():
            if position and isinstance(position, (VariantPosition, tuple)):
                positions = [position] if isinstance(position, VariantPosition) else position
                for pos in positions:
                    if isinstance(pos, VariantPosition) and pos.is_intronic():
                        seq_type = expected_prefix or 'unknown'
                        if seq_type == 'c':
                            logger.info(f"Intronic position {pos} in coding sequence - likely splice variant")
                        elif seq_type == 'n':
                            logger.info(f"Processing intronic position {pos} in non-coding sequence")
                        else:
                            logger.warning(f"Unexpected intronic position {pos} for sequence type '{seq_type}'")

    # Skip synonymous variants
    if variant.is_target_identical():
        return None

    return variant

def get_alternate_dna_sequence(dna_sequence: str, hgvs_nt: str, verbose: bool = True) -> Optional[str]:
    """Apply HGVS variant to DNA sequence using mavehgvs.

    Args:
        dna_sequence: Reference DNA sequence (can be coding, non-coding, or genomic)
        hgvs_nt: HGVS variant string (e.g., 'c.123A>T', 'n.3+2A>T', 'g.100G>A')
        verbose: Whether to log warnings for invalid variants

    Returns:
        Modified DNA sequence if variant is valid, None if invalid or fails

    Note:
        Intronic positions (e.g., +2, -3) are supported for:
        - Non-coding sequences (n.): positions map to the actual sequence
        - Coding sequences (c.): positions indicate splice sites (not applied to cDNA)
        Genomic sequences (g, m, o) use absolute positions without intronic notation.
    """
    # Handle empty/wildtype cases
    if hgvs_nt in ('_wt', '') or not hgvs_nt.strip():
        return dna_sequence

    # Skip variants with invalid nucleotides
    if 'X>' in hgvs_nt or '>X' in hgvs_nt:
        return None

    # Check for positioned synonymous variants first (handle gene-prefixed variants)
    hgvs_nt_clean = hgvs_nt.strip()
    variant_part = hgvs_nt_clean
    if ':' in hgvs_nt_clean:
        _, variant_part = hgvs_nt_clean.split(':', 1)

    if '.' in variant_part:
        prefix, suffix = variant_part.split('.', 1)
        if prefix in ['c', 'n', 'g', 'm', 'o'] and POSITIONED_EQUAL_PATTERN.match(suffix):
            return dna_sequence  # Synonymous variants return original sequence

    # Check for multi-variants that are all synonymous
    if re.match(r'^[cnhgmo]\.\[.*=.*\]$', variant_part):
        match = re.match(r'^([cnhgmo])\.\[(.+)\]$', variant_part)
        if match:
            components = match.group(2).split(';')
            non_synonymous = [comp.strip() for comp in components if not comp.strip().endswith('=')]
            if not non_synonymous:
                return dna_sequence  # All components are synonymous

    # Parse and validate variant using mavehgvs
    variant = _parse_and_validate_variant(hgvs_nt, dna_sequence, verbose=verbose)
    if variant is None:
        if verbose:
            logging.warning(f"Failed to parse or validate HGVS variant '{hgvs_nt}'")
        return None

    # Return original sequence for synonymous variants
    if variant.is_target_identical():
        return dna_sequence

    # Determine sequence type from variant prefix
    seq_type = variant.prefix if hasattr(variant, 'prefix') else None

    # Apply variant modifications using mavehgvs variant_tuples
    current_sequence = list(dna_sequence)

    for variant_type, position, sequence_data in variant.variant_tuples():
        if position is None:
            continue

        # Process based on variant type
        if variant_type == 'sub':
            # Single position substitution
            pos_idx = _extract_position_index(position, seq_type)
            if pos_idx is not None and 0 <= pos_idx < len(current_sequence):
                ref, alt = sequence_data
                current_sequence[pos_idx] = alt

        elif variant_type == 'del':
            # Deletion can be single or range
            if isinstance(position, tuple):
                start_idx = _extract_position_index(position[0], seq_type)
                end_idx = _extract_position_index(position[1], seq_type)
                if start_idx is not None and end_idx is not None:
                    if 0 <= start_idx < len(current_sequence) and 0 <= end_idx < len(current_sequence):
                        del current_sequence[start_idx:end_idx + 1]
            else:
                pos_idx = _extract_position_index(position, seq_type)
                if pos_idx is not None and 0 <= pos_idx < len(current_sequence):
                    del current_sequence[pos_idx]

        elif variant_type == 'ins':
            # Insertion happens between two positions
            if isinstance(position, tuple):
                # Insert after the end position
                end_idx = _extract_position_index(position[1], seq_type)
                if end_idx is not None and 0 <= end_idx < len(current_sequence):
                    for i, base in enumerate(sequence_data):
                        current_sequence.insert(end_idx + 1 + i, base)

        elif variant_type == 'dup':
            # Duplication can be single or range
            if isinstance(position, tuple):
                start_idx = _extract_position_index(position[0], seq_type)
                end_idx = _extract_position_index(position[1], seq_type)
                if start_idx is not None and end_idx is not None:
                    if 0 <= start_idx < len(current_sequence) and 0 <= end_idx < len(current_sequence):
                        dup_seq = current_sequence[start_idx:end_idx + 1]
                        current_sequence[end_idx + 1:end_idx + 1] = dup_seq
            else:
                pos_idx = _extract_position_index(position, seq_type)
                if pos_idx is not None and 0 <= pos_idx < len(current_sequence):
                    current_sequence.insert(pos_idx + 1, current_sequence[pos_idx])

        elif variant_type == 'delins':
            # Delete and insert at same position
            if isinstance(position, tuple):
                start_idx = _extract_position_index(position[0], seq_type)
                end_idx = _extract_position_index(position[1], seq_type)
                if start_idx is not None and end_idx is not None:
                    if 0 <= start_idx < len(current_sequence) and 0 <= end_idx < len(current_sequence):
                        current_sequence[start_idx:end_idx + 1] = list(sequence_data)
            else:
                pos_idx = _extract_position_index(position, seq_type)
                if pos_idx is not None and 0 <= pos_idx < len(current_sequence):
                    current_sequence[pos_idx] = sequence_data[0] if sequence_data else ''

    return ''.join(current_sequence)


def _get_variant_type(hgvs_variant: str) -> str:
    """Determine the variant type from HGVS notation.

    Args:
        hgvs_variant: HGVS variant string

    Returns:
        Variant type: 'sub', 'del', 'ins', 'dup', 'delins', 'synonymous', 'multi', or 'unknown'
    """
    # Handle gene-prefixed variants
    variant_part = hgvs_variant.strip()
    if ':' in hgvs_variant:
        _, variant_part = hgvs_variant.split(':', 1)

    # Check if multi-variant first (has brackets with semicolons)
    if re.match(r'^[cnhgmo]\.\[.*\]$', variant_part) and ';' in variant_part:
        return 'multi'

    # Check for specific mutation types (order matters for delins)
    if 'delins' in variant_part:
        return 'delins'
    elif 'del' in variant_part:
        return 'del'
    elif 'ins' in variant_part:
        return 'ins'
    elif 'dup' in variant_part:
        return 'dup'
    elif '>' in variant_part:
        return 'sub'  # substitution/missense
    elif '=' in variant_part:
        return 'synonymous'
    else:
        return 'unknown'


def _should_process_variant(row: pd.Series, region_type: Optional[str]) -> bool:
    """Check if variant should be processed based on filtering criteria.

    Args:
        row: DataFrame row containing variant data
        region_type: Region type filter ('coding', 'noncoding', or None)

    Returns:
        True if variant should be processed
    """
    if pd.isna(row['hgvs_nt']) or pd.isna(row['score']):
        return False

    hgvs_prefix = row['hgvs_nt'].split('.')[0]
    if ((region_type == 'coding' and hgvs_prefix != 'c') or
        (region_type == 'noncoding' and hgvs_prefix != 'n')):
        return False

    return True

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

            # Prepare metadata components once per experiment
            title = exp.get('title', '')
            description = exp.get('description', '')
            annotation = f"{title}: {description}"
            urn = exp.get('urn')
            sequence_type_str = target_genes[0]['sequence_type']

            # Filter DataFrame rows efficiently
            valid_rows = scores[scores.apply(lambda row: _should_process_variant(row, region_type), axis=1)]

            for _, row in valid_rows.iterrows():
                hgvs_nt = row['hgvs_nt']
                alt_sequence = get_alternate_dna_sequence(ref_sequence, hgvs_nt, verbose=verbose_warnings)
                if alt_sequence is None:
                    continue

                processed_count += 1
                pos_annotations = get_variant_position_annotations(hgvs_nt)
                hgvs_prefix = hgvs_nt.split('.')[0]

                # Determine variant type
                variant_type = _get_variant_type(hgvs_nt)

                position_info = f"Position Type: {pos_annotations['position_type']}"
                if pos_annotations['is_utr']:
                    position_info += ", UTR"
                if pos_annotations['is_extended']:
                    position_info += ", Extended Syntax"
                if pos_annotations['is_intronic']:
                    position_info += f", Intronic (pos: {pos_annotations['intronic_position']})"

                metadata = (f"{annotation}, Sequence Type: {sequence_type_str}, "
                           f"HGVS Prefix: {hgvs_prefix}, Variant Type: {variant_type}, "
                           f"{position_info}, URN: {urn}, HGVS: {hgvs_nt}")

                data.append([[ref_sequence, alt_sequence, metadata], row[target]])

    logger.info(f"Total studies: {study_count}/{len(urn_ids)}, MAVEs: {processed_count}/{total_count}")
    return data
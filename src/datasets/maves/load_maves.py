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
    """Make API request with retry logic and proper error handling."""
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
    """Fetch all URN IDs from MAVE-DB experiments API with error handling."""
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
    """Extract target gene information from JSON data with robust error handling."""
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
    """Fetch scores for a given URN ID with error handling."""
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
    """Fetch score set information for a given URN ID with error handling."""
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
    """Validate HGVS string using comprehensive mavehgvs patterns."""
    return HGVS_VALIDATION_PATTERN.match(hgvs_nt) is not None

def _validate_position(position) -> None:
    """Validate position using VariantPosition methods."""
    if isinstance(position, VariantPosition) and position.is_intronic():
        raise MaveHgvsParseError(f"Intronic positions not supported: {position}")

def _validate_nucleotide_sequence(seq: str, context: str) -> None:
    """Validate nucleotide sequence using dna_nt pattern."""
    if seq and not NT_PATTERN.match(seq):
        raise MaveHgvsParseError(f"Invalid nucleotide sequence '{seq}' in {context}")

def get_alternate_dna_sequence(dna_sequence: str, hgvs_nt: str, verbose: bool = True) -> Optional[str]:
    """Apply HGVS variant using mavehgvs parse_variant_strings."""
    if hgvs_nt in ('_wt', '') or not hgvs_nt.strip() or 'X>' in hgvs_nt or '>X' in hgvs_nt:
        return dna_sequence if hgvs_nt in ('_wt', '') or not hgvs_nt.strip() else None

    # Handle synonymous variants with position (e.g., n.1=, c.123=)
    # These indicate no change to the sequence
    if re.match(r'^[cngo]\.\d+=\s*$', hgvs_nt.strip()):
        return dna_sequence

    if not _validate_hgvs_format(hgvs_nt):
        raise MaveHgvsParseError(f"Invalid HGVS format: {hgvs_nt}")

    variants, errors = parse_variant_strings([hgvs_nt], targetseq=dna_sequence)
    if errors[0] or not variants[0]:
        return None

    current_sequence = list(dna_sequence)
    for variant_type, position, sequence_data in variants[0].variant_tuples():
        _validate_position(position)

        if isinstance(position, tuple):
            # For tuple positions, use the first element as the starting position
            start_pos = position[0]
            pos_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
        elif isinstance(position, VariantPosition):
            pos_idx = position.position - 1
        else:
            pos_idx = position - 1

        if variant_type == 'sub':
            ref, alt = sequence_data
            _validate_nucleotide_sequence(ref, "substitution reference")
            _validate_nucleotide_sequence(alt, "substitution alternative")
            current_sequence[pos_idx] = alt

        elif variant_type == 'del':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                del current_sequence[start_idx:end_idx]
            else:
                del current_sequence[pos_idx]

        elif variant_type == 'ins':
            _validate_nucleotide_sequence(sequence_data, "insertion")
            for i, base in enumerate(sequence_data):
                current_sequence.insert(pos_idx + 1 + i, base)

        elif variant_type == 'dup':
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                start_pos, end_pos = sequence_data
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                dup_seq = current_sequence[start_idx:end_idx]
                current_sequence[end_idx:end_idx] = dup_seq
            else:
                current_sequence.insert(pos_idx + 1, current_sequence[pos_idx])

        elif variant_type == 'delins':
            # For delins, sequence_data is just the alternate sequence string
            alt = sequence_data
            _validate_nucleotide_sequence(alt, "deletion-insertion")
            if isinstance(position, tuple):
                start_pos, end_pos = position
                start_idx = start_pos.position - 1 if isinstance(start_pos, VariantPosition) else start_pos - 1
                end_idx = end_pos.position if isinstance(end_pos, VariantPosition) else end_pos
                current_sequence[start_idx:end_idx] = list(alt)
            else:
                current_sequence[pos_idx] = alt

    return ''.join(current_sequence)





def get_maves(Seq_length=1024, limit=None, target='score', sequence_type='dna', region_type=None, verbose_warnings=True):
    """Load MAVE data with comprehensive HGVS processing."""
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
            if len(ref_sequence) > Seq_length:
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
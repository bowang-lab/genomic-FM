def filter_records(records, condition):
    """
    Filters a list of VCF records based on a given condition.

    Args:
        records (list of dict): List of VCF records, each represented as a dictionary.
        condition (function): A function that takes a record (dict) as input and returns True if the record meets the condition, False otherwise.

    Returns:
        list of dict: Filtered list of VCF records.
    """

    return [record for record in records if condition(record)]

#---------------------------------
### define condition functions ###
#---------------------------------

def is_chromosome(record, chromosome_number):
    """Check if the record is on a specific chromosome."""
    return record.get('Chromosome') == str(chromosome_number)

def is_within_position_range(record, start, end):
    """Check if the record is within a specific position range."""
    position = record.get('Position', 0)
    return start <= position <= end

def is_specific_gene(record, gene_name):
    """Check if the record is related to a specific gene."""
    gene_info = record.get('GENEINFO', '')
    return gene_name in gene_info

def is_clinical_significance(record, significance):
    """Check if the record has a specific clinical significance."""
    return significance in record.get('CLNSIG', [])

def is_allele_frequency_above(record, field, threshold):
    """Check if the allele frequency in a specific database field is above a threshold."""
    frequency = record.get(field, 'NA')
    return frequency != 'NA' and float(frequency) > threshold

def is_variant_type(record, variant_type):
    """Check if the record's variant type matches a specific type."""
    return record.get('CLNVC') == variant_type

def has_dbsnp_id(record):
    """Check if the record has a dbSNP ID."""
    return record.get('RS') != 'NA'

def is_origin_type(record, origin_type):
    """Check if the record's origin matches a specific type."""
    return origin_type in record.get('ORIGIN', [])
# ---------------------------------
### filtered_records = filter_records(vcf_records, lambda record: is_chromosome(record, 1))
# ---------------------------------

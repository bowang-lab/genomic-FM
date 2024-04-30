import subprocess
from Bio.Blast import NCBIXML
from Bio import SeqIO
from io import StringIO
import os
from kipoiseq import Interval 

def run_blast_query(sequence, fasta_path, top_match_only=True):
    # Path for the output database
    db_name = fasta_path + ".db"

    # Check if BLAST database is already created
    if not os.path.exists(db_name + ".nsq"):
        # Create a BLAST database if not already done
        subprocess.run(['makeblastdb', '-in', fasta_path, '-dbtype', 'nucl', '-out', db_name])

    # Save the query sequence to a temporary file
    query_file_path = "./root/data/query.fasta"
    with open(query_file_path, 'w') as file:
        file.write(">Query\n" + sequence)

    # Run BLAST locally
    blast_command = ['blastn', '-query', query_file_path, '-db', db_name, '-outfmt', '5']
    result = subprocess.run(blast_command, capture_output=True, text=True)

    # Parse BLAST output using StringIO to simulate a file handle
    result_handle = StringIO(result.stdout)
    blast_record = NCBIXML.read(result_handle)

    # Initialize a variable to store the best hit
    best_hsp = None
    best_alignment = None

    # Iterate through alignments to find the one with the lowest e-value
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            if top_match_only:
                if best_hsp is None or hsp.expect < best_hsp.expect:
                    best_hsp = hsp
                    best_alignment = alignment

    # Clean up temporary files
    os.remove(query_file_path)

    if best_hsp:
        chrom = best_alignment.hit_def.split("|")[0].split(' ')[0]
        start = best_hsp.sbjct_start
        end = best_hsp.sbjct_end
        return Interval(chrom, start, end)  
    else:
        return None

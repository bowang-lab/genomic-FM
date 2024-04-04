from tqdm import tqdm
import json
from collections import deque

class NCBIFastaStringExtractor:
    def __init__(self, fasta_file):
        with open(fasta_file, 'r') as f:
            self.fasta_content = f.read()
        self.lines = self.fasta_content.split('\n')
        self.sequences = {}
        self._shifts = {}  # Tracks the shift in positions for each chromosome
        self._parse_fasta()
        self.token_mapping = {
                    "<gene_start>": "B",
                    "<gene_end>": "b",
                    "<exon_start>": "D",
                    "<exon_end>": "d",
                    "<CDS_start>": "E",
                    "<CDS_end>": "e",
                    "<start_codon_start>": "F",
                    "<start_codon_end>": "f",
                    "<stop_codon_start>": "H",
                    "<stop_codon_end>": "h",
                    "<mRNA_exon_start>": "I",
                    "<mRNA_exon_end>": "i",
                    "<transcript_exon_start>": "J",
                    "<transcript_exon_end>": "j",
                    "<miRNA_exon_start>": "L",
                    "<miRNA_exon_end>": "l",
                    "<lnc_RNA_exon_start>": "M",
                    "<lnc_RNA_exon_end>": "m",
                    "<primary_transcript_exon_start>": "O",
                    "<primary_transcript_exon_end>": "o",
                    # Additional biotypes
                    "<snoRNA_exon_start>": "P",
                    "<snoRNA_exon_end>": "p",
                    "<Y_RNA_exon_start>": "Q",
                    "<Y_RNA_exon_end>": "q",
                    "<C_gene_segment_exon_start>": "R",
                    "<C_gene_segment_exon_end>": "r",
                    "<scRNA_exon_start>": "S",
                    "<scRNA_exon_end>": "s",
                    "<RNase_P_RNA_exon_start>": "*", # Note: 'T' and 't' are not reserved
                    "<RNase_P_RNA_exon_end>": "1",  # Note: 'T' and 't' are not reserved
                    "<snRNA_exon_start>": "U",
                    "<snRNA_exon_end>": "u",
                    "<tRNA_exon_start>": "V",
                    "<tRNA_exon_end>": "v",
                    "<D_gene_segment_exon_start>": "W",
                    "<D_gene_segment_exon_end>": "w",
                    "<J_gene_segment_exon_start>": "X",
                    "<J_gene_segment_exon_end>": "x",
                    "<rRNA_exon_start>": "Y",
                    "<rRNA_exon_end>": "y",
                    "<V_gene_segment_exon_start>": "Z",
                    "<V_gene_segment_exon_end>": "z",
                    "<telomerase_RNA_exon_start>": "K",
                    "<telomerase_RNA_exon_end>": "k",
                    "<ncRNA_exon_start>": "2", #
                    "<ncRNA_exon_end>": "3",   #
                    "<antisense_RNA_exon_start>": "4", #
                    "<antisense_RNA_exon_end>": "5",  # Note: 'G' and 'g' are not reserved
                    "<vault_RNA_exon_start>": "6",  # Choose alternative since 'A' is reserved
                    "<vault_RNA_exon_end>": "7",  # Choose alternative since 'a' is reserved
                    "<RNase_MRP_RNA_exon_start>": "8",  # Choose alternative since 'C' is reserved
                    "<RNase_MRP_RNA_exon_end>": "9",  # Choose alternative since 'c' is reserved
                }

    def _parse_fasta(self):
        current_header = ''
        for line in self.lines:
            if line.startswith('>'):
                current_header = line[1:].split()[0]
                self.sequences[current_header] = deque()
                self._shifts[current_header] = 0
            else:
                self.sequences[current_header].extend(line)

    def insert_token(self, chromosome, position, token, mapping=True):
        if chromosome not in self.sequences:
            raise ValueError(f"Chromosome {chromosome} not found in FASTA file.")
        if mapping:
            if token not in self.token_mapping:
                raise ValueError(f"Token {token} not found in token mapping.")
            token = self.token_mapping[token]

        adjusted_position = position + self._shifts[chromosome]
        sequence_deque = self.sequences[chromosome]

        if adjusted_position < 0 or adjusted_position > len(sequence_deque) + 1:
            raise ValueError(f"Position is out of range for chromosome {chromosome}. Position: {position}, Adjusted Position: {adjusted_position}, Sequence Length: {len(sequence_deque)}")

        sequence_deque.rotate(-adjusted_position)
        sequence_deque.extendleft(token)
        sequence_deque.rotate(adjusted_position)

        self._shifts[chromosome] += len(token)

    def save_to_file(self, output_file):
        with open(output_file, 'w') as out_f:
            for chrom, seq_deque in self.sequences.items():
                out_f.write(f'>{chrom}\n')
                out_f.write(''.join(seq_deque) + '\n')

        with open(output_file + '.json', 'w') as out_f:
            json.dump(self.token_mapping, out_f)

    def save_chrm_to_file(self, output_file, chrm):
        if chrm not in self.sequences:
            raise ValueError(f"Chromosome {chrm} not found in FASTA file.")
        output_file = output_file + f'_{chrm}.fna'
        with open(output_file, 'w') as out_f:
            out_f.write(f'>{chrm}\n')
            out_f.write(''.join(self.sequences[chrm]) + '\n')

    def update(self, chrm, sequence):
        if chrm not in self.sequences:
            raise ValueError(f"Chromosome {chrm} not found in FASTA file.")
        self.sequences[chrm] = deque(sequence)


class NCBIFastaStringExtractor_second:
    def __init__(self, fasta_file):
        with open(fasta_file, 'r') as f:
            self.fasta_content = f.read()
        self.lines = self.fasta_content.split('\n')
        self.sequences = {}
        self._shifts = {}  # Tracks the shift in positions for each chromosome
        self._parse_fasta()
        self.token_mapping = {
                    "<gene_start>": "B",
                    "<gene_end>": "b",
                    "<exon_start>": "D",
                    "<exon_end>": "d",
                    "<CDS_start>": "E",
                    "<CDS_end>": "e",
                    "<start_codon_start>": "F",
                    "<start_codon_end>": "f",
                    "<stop_codon_start>": "H",
                    "<stop_codon_end>": "h",
                    "<mRNA_exon_start>": "I",
                    "<mRNA_exon_end>": "i",
                    "<transcript_exon_start>": "J",
                    "<transcript_exon_end>": "j",
                    "<miRNA_exon_start>": "L",
                    "<miRNA_exon_end>": "l",
                    "<lnc_RNA_exon_start>": "M",
                    "<lnc_RNA_exon_end>": "m",
                    "<primary_transcript_exon_start>": "O",
                    "<primary_transcript_exon_end>": "o",
                    # Additional biotypes
                    "<snoRNA_exon_start>": "P",
                    "<snoRNA_exon_end>": "p",
                    "<Y_RNA_exon_start>": "Q",
                    "<Y_RNA_exon_end>": "q",
                    "<C_gene_segment_exon_start>": "R",
                    "<C_gene_segment_exon_end>": "r",
                    "<scRNA_exon_start>": "S",
                    "<scRNA_exon_end>": "s",
                    "<RNase_P_RNA_exon_start>": "*", # Note: 'T' and 't' are not reserved
                    "<RNase_P_RNA_exon_end>": "1",  # Note: 'T' and 't' are not reserved
                    "<snRNA_exon_start>": "U",
                    "<snRNA_exon_end>": "u",
                    "<tRNA_exon_start>": "V",
                    "<tRNA_exon_end>": "v",
                    "<D_gene_segment_exon_start>": "W",
                    "<D_gene_segment_exon_end>": "w",
                    "<J_gene_segment_exon_start>": "X",
                    "<J_gene_segment_exon_end>": "x",
                    "<rRNA_exon_start>": "Y",
                    "<rRNA_exon_end>": "y",
                    "<V_gene_segment_exon_start>": "Z",
                    "<V_gene_segment_exon_end>": "z",
                    "<telomerase_RNA_exon_start>": "K",
                    "<telomerase_RNA_exon_end>": "k",
                    "<ncRNA_exon_start>": "2", #
                    "<ncRNA_exon_end>": "3",   #
                    "<antisense_RNA_exon_start>": "4", #
                    "<antisense_RNA_exon_end>": "5",  # Note: 'G' and 'g' are not reserved
                    "<vault_RNA_exon_start>": "6",  # Choose alternative since 'A' is reserved
                    "<vault_RNA_exon_end>": "7",  # Choose alternative since 'a' is reserved
                    "<RNase_MRP_RNA_exon_start>": "8",  # Choose alternative since 'C' is reserved
                    "<RNase_MRP_RNA_exon_end>": "9",  # Choose alternative since 'c' is reserved
                }

    def _parse_fasta(self):
        current_header = ''
        for line in self.lines:
            if line.startswith('>'):
                current_header = line[1:].split()[0]
                self.sequences[current_header] = []
                self._shifts[current_header] = 0  # Initialize shift for this chromosome
            else:
                self.sequences[current_header].extend(list(line))  # Store sequence as a list of characters

    def insert_token(self, chromosome, position, token, mapping=True):
        if chromosome not in self.sequences:
            raise ValueError(f"Chromosome {chromosome} not found in FASTA file.")
        if mapping:
            if token not in self.token_mapping:
                raise ValueError(f"Token {token} not found in token mapping.")
            token = self.token_mapping[token]

        adjusted_position = position + self._shifts[chromosome]
        sequence = self.sequences[chromosome]
        if adjusted_position < 0 or adjusted_position > len(sequence) + 1:
            # print(f"Position is out of range for chromosome {chromosome}.")
            # print(f"Position: {position}, Adjusted Position: {adjusted_position}, Sequence Length: {len(sequence)}"
            # more detailed raise
            raise ValueError(f"Position is out of range for chromosome {chromosome}. Position: {position}, Adjusted Position: {adjusted_position}, Sequence Length: {len(sequence)}")
            # raise ValueError(f"Position is out of range for chromosome {chromosome}.")
        if adjusted_position == len(sequence)+1:
            sequence.append(token)
        else:
            sequence[adjusted_position:adjusted_position] = list(token)  # Insert the token
        self._shifts[chromosome] += len(token)

    def save_to_file(self, output_file):
        with open(output_file, 'w') as out_f:
            for chrom, char_list in self.sequences.items():
                out_f.write(f'>{chrom}\n')
                out_f.write(''.join(char_list) + '\n')  # Join the list into a string

        # save mapping as json
        with open(output_file + '.json', 'w') as out_f:
            import json
            json.dump(self.token_mapping, out_f)

    def save_chrm_to_file(self, output_file, chrm):
        output_file = output_file + f'_{chrm}.fna'
        with open(output_file, 'w') as out_f:
            out_f.write(f'>{chrm}\n')
            out_f.write(''.join(self.sequences[chrm]) + '\n')  # Join the list into a string

    def update(self, chrm, sequence):
        if isinstance(sequence, str):
            sequence = list(sequence)  # Convert string to list
        self.sequences[chrm] = sequence


class NCBIFastaStringExtractor_old:
    def __init__(self, fasta_file):
        with open(fasta_file, 'r') as f:
            self.fasta_content = f.read()
        self.lines = self.fasta_content.split('\n')
        self.sequences = {}
        self._shifts = {}  # Tracks the shift in positions for each chromosome
        self._parse_fasta()
        self.token_mapping = {
            "<gene_start>": "B",
            "<gene_end>": "b",
            "<exon_start>": "D",
            "<exon_end>": "d",
            "<CDS_start>": "E",
            "<CDS_end>": "e",
            "<start_codon_start>": "F",
            "<start_codon_end>": "f",
            "<stop_codon_start>": "H",
            "<stop_codon_end>": "h",
            "<mRNA_exon_start>": "I",
            "<mRNA_exon_end>": "i",
            "<transcript_exon_start>": "J",
            "<transcript_exon_end>": "j",
            "<miRNA_exon_start>": "L",
            "<miRNA_exon_end>": "l",
            "<lnc_RNA_exon_start>": "M",
            "<lnc_RNA_exon_end>": "m",
            "<primary_transcript_exon_start>": "O",
            "<primary_transcript_exon_end>": "o",
        }

    def _parse_fasta(self):
        current_header = ''
        for line in self.lines:
            if line.startswith('>'):
                current_header = line[1:].split()[0]
                self.sequences[current_header] = []
                self._shifts[current_header] = 0  # Initialize shift for this chromosome
            else:
                self.sequences[current_header].append(line)


    def insert_token(self, chromosome, position, token, mapping=True):
        if chromosome not in self.sequences:
            raise ValueError(f"Chromosome {chromosome} not found in FASTA file.")
        if mapping:
            if token not in self.token_mapping:
                raise ValueError(f"Token {token} not found in token mapping.")
            token = self.token_mapping[token]

        adjusted_position = position + self._shifts[chromosome]  # Adjust position based on current shift
        sequence = ''.join(self.sequences[chromosome])
        if adjusted_position < 0 or adjusted_position > len(sequence):
            raise ValueError(f"Position is out of range for chromosome {chromosome}.")

        new_sequence = sequence[:adjusted_position] + token + sequence[adjusted_position:]
        self.sequences[chromosome] = [new_sequence]
        self._shifts[chromosome] += len(token)  # Update the shift for this chromosome

    def save_to_file(self, output_file):
        with open(output_file, 'w') as out_f:
            for chrom, seq_list in self.sequences.items():
                out_f.write(f'>{chrom}\n')
                for seq in seq_list:
                    out_f.write(seq + '\n')
        # save mapping as json
        with open(output_file + '.json', 'w') as out_f:
            import json
            json.dump(self.token_mapping, out_f)

    def save_chrm_to_file(self, output_file, chrm):
        output_file = output_file + f'_{chrm}.fna'
        with open(output_file, 'w') as out_f:
            out_f.write(f'>{chrm}\n')
            for seq in self.sequences[chrm]:
                out_f.write(seq + '\n')

    def update(self, chrm, sequence):
        self.sequences[chrm] = sequence


def add_tokens_to_fasta_from_gtf(fasta_file, gtf_file, output_file):
    extractor = NCBIFastaStringExtractor(fasta_file)
    def get_line_count(file_path):
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    line_count = get_line_count(gtf_file)
    tokens_to_insert = []

    def extract_biotype(attributes_str):
        biotype_key = 'transcript_biotype "'
        start_index = attributes_str.find(biotype_key)
        if start_index == -1:
            return None  # Return None if biotype not found

        # Move the start index to the beginning of the actual biotype value
        start_index += len(biotype_key)
        end_index = attributes_str.find('"', start_index)
        return attributes_str[start_index:end_index] if end_index != -1 else None

    def insert_tokens_and_reset():
        for chrom, position, token in sorted(tokens_to_insert, key=lambda x: x[1]):
            extractor.insert_token(chromosome=chrom, position=position, token=token)
        tokens_to_insert.clear()

    with open(gtf_file, 'r') as gtf:
        for line in tqdm(gtf, total=line_count, desc="Processing GTF file"):
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            chrom, feature_type, start, end, attributes = parts[0], parts[2], int(parts[3]), int(parts[4]), parts[8]

            if feature_type == 'gene':
                if tokens_to_insert:
                    insert_tokens_and_reset()
                tokens_to_insert.append((chrom, start, '<gene_start>'))
                tokens_to_insert.append((chrom, end + 1, '<gene_end>'))
            elif feature_type in ['exon', 'CDS', 'start_codon', 'stop_codon']:
                if feature_type == 'exon':
                    biotype = extract_biotype(attributes)
                    token_start = f'<{biotype}_{feature_type}_start>'
                    token_end = f'<{biotype}_{feature_type}_end>'
                else:
                    token_start = f'<{feature_type}_start>'
                    token_end = f'<{feature_type}_end>'
                tokens_to_insert.append((chrom, start, token_start))
                tokens_to_insert.append((chrom, end + 1, token_end))
    if tokens_to_insert:
        insert_tokens_and_reset()

    extractor.save_to_file(output_file)

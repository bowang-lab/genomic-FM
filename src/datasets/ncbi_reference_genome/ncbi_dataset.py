from tqdm import tqdm


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

    def insert_token(self, chromosome, position, token, mapping=False):
        if mapping:
            token = self.token_mapping[token]
        if chromosome not in self.sequences:
            raise ValueError(f"Chromosome {chromosome} not found in FASTA file.")

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

    def save_chrm_to_file(self, output_file, chrm):
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
        for chrom, position, token in sorted(tokens_to_insert):
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

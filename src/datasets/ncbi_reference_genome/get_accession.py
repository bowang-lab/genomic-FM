from Bio import Entrez

Entrez.email = "vallisubasri@gmail.com"

def search_species(species_name):
    handle = Entrez.esearch(db="taxonomy", term=species_name, retmode="xml")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_species_details(tax_id):
    handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml", retmax=1)
    records = Entrez.read(handle)
    handle.close()
    return records[0]  # Assuming only one ID was queried

def get_main_assembly_accession(species, reference=True):
    try:
        if reference:
            search_query = f'"{species}"[Organism] AND (reference genome[filter] OR refseq[filter]) AND latest[filter]'
        else:
            search_query = f'"{species}"[Organism] AND latest[filter]'
        
        # Search the NCBI Assembly database
        search_handle = Entrez.esearch(db="assembly", term=search_query, retmax="100")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        refseq_assemblies = []
        genbank_assemblies = []

        # Extract the Assembly IDs from the search results
        assembly_id_list = search_results["IdList"]
        if assembly_id_list:
            for assembly_id in assembly_id_list:
                # Fetch details for each assembly found
                fetch_handle = Entrez.esummary(db="assembly", id=assembly_id, report="full")
                fetch_results = Entrez.read(fetch_handle)
                fetch_handle.close()

                for assembly in fetch_results['DocumentSummarySet']['DocumentSummary']:
                    assembly_accession = assembly['AssemblyAccession']
                    # Check if it's a RefSeq assembly by accession prefix (GCF)
                    if assembly_accession.startswith('GCF'):
                        refseq_assemblies.append(assembly_accession)
                    elif assembly_accession.startswith('GCA'):
                        genbank_assemblies.append(assembly_accession)

            # Prioritize returning a RefSeq assembly if available
            if refseq_assemblies:
                return refseq_assemblies[0]
            # Fallback to the best available GenBank assembly if no RefSeq assemblies are found
            elif genbank_assemblies:
                return genbank_assemblies[0]
            else:
                return "Suitable assemblies not found"
    except Exception as e:
        return f"Error: {str(e)}"


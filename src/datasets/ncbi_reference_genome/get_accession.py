from Bio import Entrez

Entrez.email = "zl6222@ic.ac.uk"

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
            # Formulate the search query to include the species name and filter for the latest reference genome
            search_query = f'"{species}"[Organism] AND "reference genome"[filter] AND "latest"[filter]'
        else:
            search_query = f'"{species}"[Organism] AND "latest"[filter]'
        # Search the NCBI Assembly database
        search_handle = Entrez.esearch(db="assembly", term=search_query, retmax="1")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        # Extract the Assembly ID from the search results
        assembly_id_list = search_results["IdList"]
        if assembly_id_list:
            assembly_id = assembly_id_list[0]

            # Fetch details for the first assembly found
            fetch_handle = Entrez.esummary(db="assembly", id=assembly_id, report="full")
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()
            # Extract the main assembly's accession number
            for assembly in fetch_results['DocumentSummarySet']['DocumentSummary']:
                return assembly['AssemblyAccession']
        else:
            return "Assembly not found"
    except Exception as e:
        return f"Error: {str(e)}"

#!/bin/bash

# Ensure species name is provided as an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <Species> <Accession> <Output Directory>"
    exit 1
fi

SPECIES=$1
ACCESSION=$2
OUTDIR=$3

echo "Downloading genome data for species: $SPECIES..."

if [ ! -d ${OUTDIR}/${SPECIES} ] ; then
    mkdir ${OUTDIR}/${SPECIES}
fi


datasets download genome accession ${ACCESSION} --dehydrated --include genome,rna,cds,protein,gtf --filename ${OUTDIR}/${SPECIES}/${ACCESSION}.zip 
unzip -o ${OUTDIR}/${SPECIES}/${ACCESSION}.zip -d ${OUTDIR}/${SPECIES}/${ACCESSION}
datasets rehydrate --directory ${OUTDIR}/${SPECIES}/${ACCESSION}

echo "Download complete."

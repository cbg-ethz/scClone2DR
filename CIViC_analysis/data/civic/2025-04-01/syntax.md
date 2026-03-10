# 29.04.25

Download from civic database, see `RESOURCES` -> `Data releases`
- "Evidence" -> `01-Apr-2025-ClinicalEvidenceSummaries.tsv`
- "Molecular Profiles" -> `01-Apr-2025-MolecularProfileSummaries.tsv`
- "Variants" -> `01-Apr-2025-VariantSummaries.tsv`


* reformat information from all files
* use custom Python script
```
python ../../../scripts/get_gene_lists.py ./ > 01-Apr-2025_per_drug_gene_lists.tsv
```

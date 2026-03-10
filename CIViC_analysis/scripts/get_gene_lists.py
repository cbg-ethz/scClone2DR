import pandas as pd
import os
import argparse
from typing import List, Optional, Dict
import glob


def find_file_with_prefix(directory, suffix):
    matches = glob.glob(os.path.join(directory, f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file ending with '{suffix}', but found: {matches}")
    return matches[0]


def load_civic_assertions(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t')

def load_variant_summaries(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t')

def load_molecular_profiles(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t')

def filter_assertions(
    df: pd.DataFrame,
    diseases: Optional[List[str]] = None,
    min_confidence_rating: Optional[int] = None,
    evidence_levels: Optional[List[str]] = None,
    variant_types: Optional[List[str]] = None
) -> pd.DataFrame:
    if diseases:
        disease_pattern = '|'.join(diseases)
        df = df[df['disease'].str.contains(disease_pattern, case=False, na=False)]
    if min_confidence_rating:
        df = df[df['rating'] >= min_confidence_rating]
    if evidence_levels:
        df = df[df['evidence_level'].isin(evidence_levels)]
    if variant_types:
        df = df[df['molecular_profile'].str.contains('|'.join(variant_types), case=False, na=False)]
    return df

def extract_gene_lists_by_drug(df: pd.DataFrame, variant_df: pd.DataFrame, molecular_profiles_df: pd.DataFrame) -> Dict[str, List[str]]:
    variant_id_to_gene = variant_df.set_index('variant_id')['gene'].to_dict()
    profile_to_variants = molecular_profiles_df.set_index('molecular_profile_id')['variant_ids'].dropna().to_dict()

    drug_gene_map = {}
    for _, row in df.iterrows():
        drugs = row['therapies'].split(',') if pd.notna(row['therapies']) else []
        profile_id = row['molecular_profile_id'] if 'molecular_profile_id' in row else None

        genes = []
        if profile_id in profile_to_variants:
            variant_ids = [int(v.strip()) for v in str(profile_to_variants[profile_id]).split(',')]
            genes = [variant_id_to_gene.get(vid) for vid in variant_ids if isinstance(variant_id_to_gene.get(vid), str)]

        for drug in drugs:
            drug = drug.strip()
            if drug:
                drug_gene_map.setdefault(drug, set()).update(genes)
    return {k: sorted(v) for k, v in drug_gene_map.items()}

def get_relevant_genes_per_drug(
    assertions_file: str,
    variant_file: str,
    molecular_profiles_file: str,
    diseases: Optional[List[str]] = None,
    min_confidence_rating: Optional[int] = None,
    evidence_levels: Optional[List[str]] = None,
    variant_types: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    df = load_civic_assertions(assertions_file)
    variant_df = load_variant_summaries(variant_file)
    molecular_profiles_df = load_molecular_profiles(molecular_profiles_file)
    filtered_df = filter_assertions(df, diseases, min_confidence_rating, evidence_levels, variant_types)
    return extract_gene_lists_by_drug(filtered_df, variant_df, molecular_profiles_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse CIViC Clinical Evidence TSV for drug-gene associations")
    parser.add_argument("resource_path", help="Path to the folder containing CIViC TSV files")
    parser.add_argument("--diseases", nargs='*', help="List of diseases to filter for", default=None)
    parser.add_argument("--min_confidence_rating", type=int, help="Minimum confidence rating", default=None)
    parser.add_argument("--evidence_levels", nargs='*', help="List of evidence levels to filter for", default=None)
    parser.add_argument("--variant_types", nargs='*', help="List of variant types to filter for", default=None)
    args = parser.parse_args()

    #assertions_path = os.path.join(args.resource_path, 'nightly-ClinicalEvidenceSummaries.tsv')
    #variant_path = os.path.join(args.resource_path, 'nightly-VariantSummaries.tsv')
    #molecular_profiles_path = os.path.join(args.resource_path, 'nightly-MolecularProfileSummaries.tsv')

    assertions_path = find_file_with_prefix(args.resource_path, 'ClinicalEvidenceSummaries.tsv')
    variant_path = find_file_with_prefix(args.resource_path, 'VariantSummaries.tsv')
    molecular_profiles_path = find_file_with_prefix(args.resource_path, 'MolecularProfileSummaries.tsv')


    genes_by_drug = get_relevant_genes_per_drug(
        assertions_file=assertions_path,
        variant_file=variant_path,
        molecular_profiles_file=molecular_profiles_path,
        diseases=args.diseases,
        min_confidence_rating=args.min_confidence_rating,
        evidence_levels=args.evidence_levels,
        variant_types=args.variant_types
    )

    for drug, genes in genes_by_drug.items():
        print(f"{drug}\t{','.join(genes)}")

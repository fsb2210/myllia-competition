import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


# from here: https://davetang.org/muse/2024/11/14/ensembl-gene-ids-to-gene-symbols/
def parse_ensembl_map(path: str) -> pd.Series:
    """
    Parse the Ensembl v112 gene table into a Series
    """
    df = pd.read_csv(path, sep="\t", header=None, usecols=[6, 7], names=["ensembl_id", "gene_name"], compression="gzip")
    df = df.dropna().drop_duplicates("ensembl_id")
    return df.set_index("ensembl_id")["gene_name"]

def resolve_gene_index(var: pd.DataFrame, genes: pd.Index, ensembl_map: pd.Series) -> pd.DataFrame:
    """
    Build a mapping table that links each gene to the integer column index it occupies in adata.X (if present)
    """
    # lookup: gene_symbol -> integer position in var
    try:
        symbol_to_idx = pd.Series(np.arange(len(var)), index=var["gene_name"].values)
    except Exception:
        symbol_to_idx = pd.Series(np.arange(len(var)), index=var["features"].values)

    # Ensembl-based lookup using the v112 map
    var_ensembl_to_symbol = ensembl_map.reindex(var.index)   # ENSG -> v112 symbol
    ensembl_symbol_to_idx = pd.Series(np.arange(len(var)), index=var_ensembl_to_symbol.values).dropna()
    records = []
    for gene in genes:
        if gene in symbol_to_idx.index:
            records.append({"output_gene": gene, "var_idx": int(symbol_to_idx[gene])})
        elif gene in ensembl_symbol_to_idx.index:
            records.append({"output_gene": gene, "var_idx": int(ensembl_symbol_to_idx[gene])})
        else:
            records.append({"output_gene": gene, "var_idx": np.nan})

    result = pd.DataFrame(records)
    n_found = result["var_idx"].notna().sum()
    n_alias = sum(1 for g, r in zip(genes, records) if not pd.isna(r["var_idx"]) and g not in symbol_to_idx.index)
    print(f"  Gene resolution:")
    print(f"    Direct symbol match : {n_found - n_alias}")
    print(f"    Ensembl alias match : {n_alias}")
    print(f"    Absent (will be NaN): {result['var_idx'].isna().sum()}")
    return result


def main(h5ad_path: str, training_csv: str, ensembl_map_path: str, pert_col: str = "gene", umi_col: str = "UMI_count") -> pd.DataFrame:
    # load reference gene + perturbation lists
    print("Loading training_data_means.csv ...")
    t_df = pd.read_csv(training_csv)
    output_genes = pd.Index(t_df.columns[1:])
    perts = pd.Index(t_df["pert_symbol"].unique())
    print(f"  Output genes : {len(output_genes)}")
    print(f"  Challenge perts : {len(perts)}")

    # load Ensembl mapping
    print("Loading Ensembl v112 gene map ...")
    ensembl_map = parse_ensembl_map(ensembl_map_path)
    print(f"  Ensembl entries : {len(ensembl_map)}")

    # open h5ad in backed mode
    print(f"\nOpening h5ad in backed mode: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)#, backed="r")
    print(f"  Single-cell shape : {adata.shape}  (cells × genes)")

    if pert_col not in adata.obs.columns:
        raise ValueError(f"Perturbation column '{pert_col}' not found. Available: {list(adata.obs.columns)}")
    # if umi_col not in adata.obs.columns:
    #     raise ValueError(f"UMI count column '{umi_col}' not found. Available: {list(adata.obs.columns)}")

    # resolve gene indices
    print("\nResolving gene indices ...")
    gene_map = resolve_gene_index(adata.var, output_genes, ensembl_map)

    present = gene_map.dropna(subset=["var_idx"])
    var_idx = present["var_idx"].astype(int).values

    # load only needed gene columns from disk
    # NOTE: backed h5ad requires column indices to be in strictly increasing order
    sort_order = np.argsort(var_idx)
    var_idx_sorted = var_idx[sort_order]
    output_genes_sorted = present["output_gene"].values[sort_order]

    # per-cell normalisation
    print("Normalising (counts / UMI_count × 10k → log2(x+1)) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)

    # get matrix with data
    X_sub = adata.X[:, var_idx_sorted]
    X_norm = X_sub.toarray() if issparse(X_sub) else X_sub # np.array(X_sub, dtype=float)
    adata.file.close()

    # average per perturbation
    print("Averaging per perturbation ...")
    pert_labels = adata.obs[pert_col].values.astype(str)
    norm_df = pd.DataFrame(X_norm, index=pert_labels, columns=output_genes_sorted)
    norm_df.index.name = "pert_symbol"
    per_pert = norm_df.groupby("pert_symbol").mean()

    # reindex to full challenge gene set (NaN for absent genes)
    per_pert = per_pert.reindex(columns=output_genes)
    per_pert.index.name = "pert_symbol"
    return per_pert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build perturbation feature matrix from a Perturb-seq h5ad file")
    parser.add_argument("--h5ad", required=True, help="Path to single-cell h5ad file")
    parser.add_argument("--training", required=True, help="Path to training_data_means.csv")
    parser.add_argument("--ensembl_map", required=True, help="Path to Ensembl v112 gene table (.gz file)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--pert_col", default="gene", help="obs column with perturbation labels (default: 'gene')")
    parser.add_argument("--umi_col", default="UMI_count", help="obs column with per-cell UMI totals (default: 'UMI_count')")
    args = parser.parse_args()

    result = main(h5ad_path=args.h5ad, training_csv=args.training, ensembl_map_path=args.ensembl_map, pert_col=args.pert_col, umi_col=args.umi_col)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output)
    print(f"\nSaved: {args.output}  shape={result.shape}")
    print(f"NaN genes (absent from h5ad): {result.isna().any(axis=0).sum()} / {result.shape[1]}")



import h5py.tests.test_file_alignment
import argparse
import time
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import scanpy as sc
import gc
from tqdm.auto import tqdm

FDR_ALPHA = 0.05
CELL_THRESHOLD = 10

def main(opts: argparse.Namespace):

    dataset_name = opts.h5ad.split(".h5ad")[0].split("/")[-1]
    print(f"\n{"="*50}")
    print(f"- `{dataset_name}` dataset")
    print(f"{"="*50}")

    # load training data
    t_df = pd.read_csv(opts.training)
    v_df = pd.read_csv(opts.valid)
    all_target_names = t_df.columns[1:].tolist()
    chall_pert_genes = t_df["pert_symbol"].tolist()[:-1]
    chall_pert_genes.extend(v_df["pert"].tolist())

    # load h5ad
    adata_path = opts.h5ad
    print(f"- loading `{adata_path}`")
    adata = sc.read_h5ad(adata_path)
    print(f"\r- preprocessing data", end="... ")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)
    print("done!")

    # find available target genes in dataset
    var_name = "gene_name"
    try:
        target_mask = adata.var[var_name].isin(all_target_names)
    except KeyError:
        var_name = "features"
        target_mask = adata.var[var_name].isin(all_target_names)
    target_idx = np.where(target_mask)[0]
    available_genes = adata.var.loc[target_mask, var_name].values
    print(f"- found {len(available_genes)}/{len(all_target_names)} target genes")

    # get names of perturbated genes
    obs_name = "gene" if "gene" in adata.obs.keys() else "sgrna_symbol"
    all_obs_names = adata.obs[obs_name].unique().tolist()
    obs_pert_genes = [name for name in all_obs_names if name != "non-targeting"]
    print(f"- have {len(obs_pert_genes)} perturbated genes")

    # if we need neutral values, load file
    COMPUTE_NEUTRAL = args.compute_neutral
    if not COMPUTE_NEUTRAL:
        t_sig_df = pd.read_csv(args.neutral_fname)
        training_gene_means = t_sig_df.groupby("gene_name")["logFC"].mean().to_dict()

    # control cells (non-targeting)
    ctrl_cells = np.where(adata.obs[obs_name] == "non-targeting")[0]

    # neutral template for missing perturbations
    neutral_df = pd.DataFrame({
        "pval": 1.0,
        "pval_adj": 1.0,
        "score": 0.0,
        "logFC": 0.0
    }, index=all_target_names)

    # results dict for 80 challenge perturbations
    results = {}
    pbar = tqdm(enumerate(chall_pert_genes), desc="scRNA gene significance", total=len(chall_pert_genes))
    for k, pert in pbar:

        if pert not in obs_pert_genes:
            # perturbation missing entirely → use neutral values
            results[pert] = neutral_df.copy()
            pbar.set_postfix({"gene": f"{pert} (missing in {dataset_name})"})
            continue

        # get perturbated cells for specific `pert`
        pert_cells = np.where(adata.obs[obs_name] == pert)[0]

        # low number of cells to make statistical analysis
        if len(pert_cells) < CELL_THRESHOLD:
            # perturbation missing entirely → use neutral values
            results[pert] = neutral_df.copy()
            pbar.set_postfix({"gene": f"{pert} (number of cells ({len(pert_cells)}) below threshold ({CELL_THRESHOLD}))"})
            continue

        pert_adata = adata[pert_cells, target_idx]
        ctrl_adata = adata[ctrl_cells, target_idx]

        # get dense expression matrices
        pert_expr = pert_adata.X.toarray() if hasattr(pert_adata.X, "toarray") else pert_adata.X
        ctrl_expr = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else ctrl_adata.X

        # Wilcoxon rank-sum test for all available genes
        statistics, pvals = ranksums(pert_expr, ctrl_expr, axis=0, alternative="two-sided")
        pvals_adj = multipletests(pvals, method="fdr_bh")[1]

        result = pd.DataFrame({
            "pval": pvals,
            "pval_adj": pvals_adj,
            "score": statistics,
            "logFC": np.mean(pert_expr, axis=0) - np.mean(ctrl_expr, axis=0)
        }, index=available_genes)

        # in case of missing genes, add dummy values
        if len(available_genes) != len(all_target_names):
            # fill missing genes with neutral values
            missing_genes = np.setdiff1d(all_target_names, available_genes)
            for gene in missing_genes:
                result.loc[gene, "logFC"] = training_gene_means.get(gene, 0.0)
                result.loc[gene, "pval"] = 1.0
                result.loc[gene, "pval_adj"] = 1.0
                result.loc[gene, "score"] = 0.0

            result = result.loc[all_target_names]

        # print some output in the terminal
        significant = pvals_adj < FDR_ALPHA
        pbar.set_postfix({"gene": f"{pert}", "significant": f"{significant.sum()}", "frac": f"{significant.sum()/len(all_target_names):.2f}"})

        # add perturbation to results dictionary
        results[pert] = result

    # build final dataframe with MultiIndex
    de_matrix = pd.concat(results.values(), keys=chall_pert_genes, names=["perturbation", "gene_name"])

    # garbage collector
    del adata
    gc.collect()

    return de_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build expression comparison matrix of Wilcoxon test results from a Perturb-seq h5ad file")
    parser.add_argument("--h5ad", required=True, help="Path to single-cell h5ad file")
    parser.add_argument("--training", required=True, help="Path to training_data_means.csv")
    parser.add_argument("--valid", required=True, help="Path to pert_ids_all.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--compute_neutral", help="Whether to compute stats for training cells", action="store_true", default=False)
    parser.add_argument("--neutral_fname", help="Filename with neutral data")
    args = parser.parse_args()

    # process h5ad file
    de_df = main(args)

    # save output to file
    if args.output: de_df.to_csv(args.output)

    print(f"- dataset ready: {de_df.shape} ({de_df.memory_usage(deep=True).sum()/1e6:.1f} MB)")

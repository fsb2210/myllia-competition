# TODO
---

- [x] use "&" and "|" instead of for-loops when filtering datasets
- [x] add self-loop (I, identity matrix) to the adjacency matrix (A) to avoid nodes with no edges

- [ ] use multi-scale diffusion: instead of a single diffusion vector with a single `alpha_RW`, concatenate multiple of them to have local + global context
        ```python
        # Define your alphas (Local, Mid-range, Global)
        alphas_RW = [0.1, 0.5, 0.9]

        all_X_train = []
        all_X_val = []

        for alpha in alphas_RW:
            print(f" - alpha = {alpha}")
            # 1. Create operator M for this alpha
            M = (1/(1 - alpha)) * (sp.eye(n_nodes) - alpha * A.dot(D_inv))
            M_csc = M.tocsc()

            # 2. LU Decomposition for this specific alpha
            LU = sp.linalg.splu(M_csc)

            # 3. Diffuse Train
            X_alpha = np.vstack([diffuse_gene(g, gene_to_idx, LU) for g in train_pert_genes])

            # 4. Diffuse Val
            X_val_alpha = np.vstack([diffuse_gene(g, gene_to_idx, LU) for g in val_pert_genes])

            all_X_train.append(X_alpha)
            all_X_val.append(X_val_alpha)

        # Concatenate features: Shape becomes (80, n_nodes * 3)
        X = np.hstack(all_X_train)
        X_val = np.hstack(all_X_val)

        print(f"New X shape: {X.shape}")
        ```

- [x] quantile-based selection of features: avoid using the entire networks and use only those genes that show variance between perturbations
- [ ] try *without* quantile-based features selection

- [ ] use `StandardScaler` for `ElasticNet` models
- [x] try different values of `n_latent`: 50, 70 -> **nothing changed**
- [ ] use AutoEncoder instead of PCA
- [ ] make predictions using an ensemble of models: `Ridge`, (multitask) `ElasticNet`
- [x] given the small sample set use *leave-one-out* (LOO) cross-validation for a less biased estimate

- [x] add visualizations for model performance: scatter plot, histogram of correlations, heatmaps

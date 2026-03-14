import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

def extract_activations(model: HookedTransformer, dataloader, layer: int = -1):
    model.eval()
    all_activations = []
    all_labels = []
    
    if layer == -1:
        layer = model.cfg.n_layers - 1
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    with torch.no_grad():
        for batch_x, _, labels in dataloader:
            batch_x = batch_x.to(model.cfg.device)
            
            _, cache = model.run_with_cache(batch_x, names_filter=[hook_name])
            
            # Extract activations: shape (batch_size, seq_len, d_model)
            acts = cache[hook_name].cpu().numpy()
            
            final_pos_acts = acts[:, -1, :] 
            
            all_activations.append(final_pos_acts)
            all_labels.append(labels.numpy())
            
    return np.vstack(all_activations), np.concatenate(all_labels)

def calculate_cev(activations: np.ndarray, max_components: int = None):
    if max_components is None:
        max_components = min(activations.shape[0], activations.shape[1])
        
    pca = PCA(n_components=max_components)
    pca.fit(activations)
    
    cev = np.cumsum(pca.explained_variance_ratio_)
    return pca, cev

def plot_cev(cev: np.ndarray, threshold: float = 0.95):
    dims_needed = np.argmax(cev >= threshold) + 1
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cev) + 1), cev, marker='o', markersize=4, linestyle='-', color='b')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% Variance')
    plt.axvline(x=dims_needed, color='g', linestyle='--', label=f'{dims_needed} Dimensions')
    
    plt.title("Cumulative Explained Variance (CEV)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dims_needed

def plot_2d_pca(activations: np.ndarray, labels: np.ndarray, pca_model: PCA, pc_x: int = 0, pc_y: int = 1):
    proj = pca_model.transform(activations)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(proj[:, pc_x], proj[:, pc_y], c=labels, cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Process ID')
    plt.title(f"Residual Stream Geometry (PC{pc_x+1} vs PC{pc_y+1})")
    plt.xlabel(f"Principal Component {pc_x+1}")
    plt.ylabel(f"Principal Component {pc_y+1}")
    plt.grid(True, alpha=0.3)
    plt.show()
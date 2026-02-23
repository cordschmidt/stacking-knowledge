import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def calculate_and_save_layer_similarity_plot(model, output_dir, step: int, stage_name: str = None):
    """
    Calculates cosine similarity between layer weights and saves the matrix as svg and png files
    """
    try:
        save_path_svg = prepare_save_path(output_dir=output_dir, stage_name=stage_name, step=step)

        layer_weights = extract_weights_for_each_llama_layer(model=model, save_path_svg=save_path_svg)

        similarity_matrix = calculate_similarity_matrix(layer_weights=layer_weights)

        save_path_svg = prepare_save_path(output_dir=output_dir, stage_name=stage_name, step=step)
        
        create_and_save_similarity_plot(similarity_matrix=similarity_matrix, save_path_svg=save_path_svg, stage_name=stage_name, step=step)

        logger.info(f"Saved layer similarity matrix to {save_path_svg}")

    except Exception as e:
        logger.error(f"Failed to calculate layer similarity at step {step}: {e}")

def extract_weights_for_each_llama_layer(model, save_path_svg):
    # Extract layers
    layers = model.model.layers if hasattr(model, "model") else model.layers
    # torch.save(layers, save_path_svg.replace(".svg", ".pt"))
    # We target the 'gate_proj' as the 'first feedforward layer'
    layer_weights = []

    for layer in layers:
        # Flatten weight: [out_features, in_features] -> [out_features * in_features]
        w = layer.mlp.gate_proj.weight.detach().cpu()
        layer_weights.append(w.view(-1))

    return layer_weights


def create_and_save_similarity_plot(similarity_matrix, save_path_svg, stage_name, step):
    # Plot layer similarities
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap='BuPu',
        vmin=0,
        vmax=1,
        annot_kws={"size": 8}
    )
    if stage_name is not None:
        plt.title(f"Layer Similarity - Stage: {stage_name} | Step: {step}")
    else:
        plt.title(f"Layer Similarity - Step: {step}")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")

    # Save as SVG and PNG for quick previews
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    plt.savefig(save_path_svg.replace(".svg", ".png"), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_similarity_matrix(layer_weights):
    weights_tensor = torch.stack(layer_weights)
    weights_norm = F.normalize(weights_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(weights_norm, weights_norm.t()).numpy()
    return similarity_matrix


def prepare_save_path(output_dir, stage_name, step):
    os.makedirs(output_dir, exist_ok=True)
    if stage_name is not None:
        filename = f"layer_similarity_stage_{stage_name}_step_{step}.svg"
    else:
        filename = f"layer_similarity_step_{step}.svg"
    save_path = os.path.join(output_dir, filename)
    return save_path
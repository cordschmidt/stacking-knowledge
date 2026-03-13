import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def create_layer_and_block_similarity_plots(model, output_dir, step: int, stage_name: str = None, block_size: int = 1):
    """
    Calculates cosine similarity between layer weights (and additionally block weights if block_size > 1)
    and saves the matrices as svg and png files
    """
    # Create plots for layer similarity
    calculate_and_save_similarity_plot(model = model, output_dir = output_dir, step = step, stage_name = stage_name, block_size = block_size, similarity_objective = "layer")

    # If block_size > 1, group the layer weights and plot additionally block-level similarity
    if block_size > 1:
        calculate_and_save_similarity_plot(model = model, output_dir = output_dir, step = step, stage_name = stage_name, block_size = block_size, similarity_objective = "block")

def calculate_and_save_similarity_plot(model, output_dir, step: int, stage_name: str = None, block_size: int = 1, similarity_objective: str = "layer"):
    """
    Calculates cosine similarity between layer weights (and additionally block weights if block_size > 1)
    and saves the matrices as svg and png files
    """
    weights = extract_weights_for_each_llama_layer(model=model)
    if similarity_objective == "block":
        weights = group_weights_into_blocks(weights, block_size)

    similarity_matrix = calculate_similarity_matrix(layer_weights=weights)

    layer_save_path = prepare_save_path(output_dir, stage_name, step, unit_name=similarity_objective)
    create_and_save_similarity_plot(
        similarity_matrix=similarity_matrix,
        save_path_svg=layer_save_path,
        stage_name=stage_name,
        step=step,
        unit_name=similarity_objective
    )
    logger.info(f"Saved {similarity_objective} similarity matrix to {layer_save_path}")

def extract_weights_for_each_llama_layer(model):
    # Extract layers safely
    layers = model.model.layers if hasattr(model, "model") else model.layers

    layer_weights = []
    for layer in layers:
        w = layer.mlp.gate_proj.weight.detach().cpu()
        layer_weights.append(w.view(-1))

    return layer_weights

def group_weights_into_blocks(layer_weights, block_size: int):
    block_weights = []
    # Group the layer weights into blocks of 'block_size'
    for i in range(0, len(layer_weights), block_size):
        # Concatenate the flattened layer vectors in this block into one giant block vector
        block = torch.cat(layer_weights[i : i + block_size])
        block_weights.append(block)
    return block_weights


def create_and_save_similarity_plot(similarity_matrix, save_path_svg, stage_name, step, unit_name="layer"):
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
        plt.title(f"{unit_name} Similarity - Stage: {stage_name} | Step: {step}")
    else:
        plt.title(f"{unit_name} Similarity - Step: {step}")

    plt.xlabel(f"{unit_name} Index")
    plt.ylabel(f"{unit_name} Index")

    # Save as SVG and PNG for quick previews
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    plt.savefig(save_path_svg.replace(".svg", ".png"), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_similarity_matrix(layer_weights):
    weights_tensor = torch.stack(layer_weights)
    weights_norm = F.normalize(weights_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(weights_norm, weights_norm.t()).numpy()
    return similarity_matrix


def prepare_save_path(output_dir, stage_name, step, unit_name="layer"):
    os.makedirs(output_dir, exist_ok=True)
    if stage_name is not None:
        filename = f"{unit_name}_similarity_stage_{stage_name}_step_{step}.svg"
    else:
        filename = f"{unit_name}_similarity_step_{step}.svg"
    save_path = os.path.join(output_dir, filename)
    return save_path
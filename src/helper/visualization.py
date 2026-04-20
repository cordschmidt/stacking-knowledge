import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def create_layer_and_block_similarity_plots(model, output_dir, step: int, stage_name: str = None, block_size: int = 1):
    """
    Calculates cosine similarity between layer weights (and block weights if block_size > 1)
    and saves the matrices as svg and png files

    Args:
        model: The model
        output_dir: The directory where the plots should be saved
        step: The current global training step
        stage_name: An optional string representing the current curriculum stage
        block_size: The integer number of layers that constitute a single architectural block

    Returns:
        None
    """
    # Create plots for layer similarity
    calculate_and_save_similarity_plot(model = model, output_dir = output_dir, step = step, stage_name = stage_name, block_size = block_size, similarity_objective = "layer")

    # If block_size > 1, group the layer weights and plot additionally block-level similarity
    if block_size > 1:
        calculate_and_save_similarity_plot(model = model, output_dir = output_dir, step = step, stage_name = stage_name, block_size = block_size, similarity_objective = "block")

def calculate_and_save_similarity_plot(model, output_dir, step: int, stage_name: str = None, block_size: int = 1, similarity_objective: str = "layer"):
    """
    Calculates and saves the cosine similarity matrix for either individual layers or architectural blocks.

    Args:
        model: The model
        output_dir: The directory where the plot should be saved
        step: The current global training step
        stage_name: An optional string representing the current curriculum stage
        block_size: The integer number of layers per block
        similarity_objective: A string ("layer" or "block") determining the level of analysis

    Returns:
        None
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
    """
    Extracts and flattens the weights from the MLP gate projection of each layer in a LLaMA model

    Args:
        model: The LLaMA model instance

    Returns:
        A list of flattened 1D PyTorch tensors, each containing the gate projection weights of a layer
    """
    # Extract layers safely
    layers = model.model.layers if hasattr(model, "model") else model.layers

    layer_weights = []
    for layer in layers:
        w = layer.mlp.gate_proj.weight.detach().cpu()
        layer_weights.append(w.view(-1))

    return layer_weights

def group_weights_into_blocks(layer_weights, block_size: int):
    """
    Concatenates individual layer weights into larger block-level weight vectors based on the specified block size

    Args:
        layer_weights: A list of 1D PyTorch tensors representing individual layer weights
        block_size: The integer number of layers to group together into a single block

    Returns:
        A list of 1D PyTorch tensors representing the concatenated block weights
    """
    block_weights = []
    # Group the layer weights into blocks of 'block_size'
    for i in range(0, len(layer_weights), block_size):
        # Concatenate the flattened layer vectors in this block into one giant block vector
        block = torch.cat(layer_weights[i : i + block_size])
        block_weights.append(block)
    return block_weights


def create_and_save_similarity_plot(similarity_matrix, save_path_svg, stage_name, step, unit_name="layer"):
    """
    Generates a heatmap from the similarity matrix using Seaborn and saves it to disk in both SVG and PNG formats

    Args:
        similarity_matrix: A 2D numpy array containing the pairwise cosine similarity scores
        save_path_svg: The full file path string for saving the SVG plot
        stage_name: The string name of the current curriculum stage (used for the plot title)
        step: The integer representing the current training step (used for the plot title)
        unit_name: A string ("layer" or "block") used for labeling the axes and title

    Returns:
        None
    """
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
    """
    Computes the pairwise cosine similarity matrix for a given list of weight vectors

    Args:
        layer_weights: A list of 1D PyTorch tensors representing the weights to be compared

    Returns:
        A 2D numpy array representing the cosine similarity matrix
    """
    weights_tensor = torch.stack(layer_weights)
    weights_norm = F.normalize(weights_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(weights_norm, weights_norm.t()).numpy()
    return similarity_matrix


def prepare_save_path(output_dir, stage_name, step, unit_name="layer"):
    """
    Constructs the full file path for saving the plot and ensures the target directory exists

    Args:
        output_dir: The base directory string where the plots should be saved
        stage_name: The optional string name of the current curriculum stage to include in the filename
        step: The integer representing the current training step to include in the filename
        unit_name: A string ("layer" or "block") to prefix the filename

    Returns:
        A string representing the full absolute or relative save path
    """
    os.makedirs(output_dir, exist_ok=True)
    if stage_name is not None:
        filename = f"{unit_name}_similarity_stage_{stage_name}_step_{step}.svg"
    else:
        filename = f"{unit_name}_similarity_step_{step}.svg"
    save_path = os.path.join(output_dir, filename)
    return save_path
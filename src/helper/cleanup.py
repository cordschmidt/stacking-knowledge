import os
import logging

logger = logging.getLogger(__name__)


def cleanup_output_dir(output_dir):
    """
    Removes redundant model files from the root output directory
    while preserving checkpoint folders
    """
    files_to_remove = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json"
        "tokenizer_config.json",
        "training_args.bin",
    ]

    logger.info(f"Cleaning up redundant files in {output_dir}...")

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)

        # Remove specific redundant files
        if filename in files_to_remove:
            try:
                os.remove(file_path)
                logger.debug(f"Removed: {filename}")
            except OSError as e:
                logger.error(f"Error deleting {file_path}: {e}")
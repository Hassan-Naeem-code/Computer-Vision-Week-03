"""File I/O utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_dir: str, name: str = "results") -> None:
    """Save results to JSON and pickle files.
    
    Args:
        results: Dictionary of results to save
        output_dir: Directory to save results
        name: Name of result file (without extension)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = Path(output_dir) / f"{name}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Results saved to {json_path}")
    except Exception as e:
        logger.warning(f"Could not save as JSON: {e}")

    # Save as pickle
    pickle_path = Path(output_dir) / f"{name}.pkl"
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {pickle_path}")
    except Exception as e:
        logger.error(f"Could not save as pickle: {e}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON or pickle file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dictionary of loaded results
    """
    results_path = Path(results_path)

    if results_path.suffix == ".json":
        with open(results_path, "r") as f:
            results = json.load(f)
    elif results_path.suffix == ".pkl":
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")

    logger.info(f"Results loaded from {results_path}")
    return results

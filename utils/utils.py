import pandas as pd
import logging
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def save_summary(provider_results: Dict, output_dir: str):
    """Save evaluation results to CSV files.

    Args:
        provider_results: Dictionary of provider results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = f"{output_dir}/summary.csv"

    with open(summary_file, 'w', newline='') as csvfile:
        fieldnames = ['provider', 'accuracy', 'correct_count', 'total_count', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for provider_name, result in provider_results.items():
            output_file = f"{output_dir}/{provider_name}_results.csv"
            provider_full_results = pd.read_csv(output_file)
            examples_count = len(provider_full_results)
            correct_count = len(provider_full_results[provider_full_results['is_correct'] == True])

            accuracy = correct_count / examples_count if examples_count > 0 else 0.0
            accuracy = round(accuracy, 3)

            writer.writerow({
                'provider': provider_name,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': examples_count,
                'timestamp': timestamp
            })

    logger.info(f"Saved summary results to {summary_file}")


def load_csv_data(
    csv_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
) -> List[Dict]:
    """Load data from CSV file with question and answer columns.

    Args:
        csv_path: Path to the CSV file
        start_index: Starting index for examples (inclusive)
        end_index: Ending index for examples (exclusive), defaults to the end of the dataset
        random_sample: Number of random samples to select (overrides start_index and end_index)
        rerun: Whether to rerun evaluation on existing results directory, output_dir must exist
        results_dir: Directory to save results
        provider_names: List of provider names to include in the results
    Returns:
        List of dictionaries with question, answer, and index keys
    """
    try:
        logger.info(f"Loading data from csv file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check if the required columns exist
        required_cols = ['problem', 'answer']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain '{col}' column")

        total_rows = len(df)

        # Add original index as a column
        df['index'] = range(len(df))

        if random_sample is not None and random_sample > 0:
            # Random sampling mode
            sample_size = min(random_sample, total_rows)
            logger.info(f"Randomly sampling {sample_size} examples from {total_rows} total")
            df_slice = df.sample(sample_size)
        else:
            # Sequential slice mode
            if end_index is None:
                end_index = total_rows

            # Ensure indices are within bounds
            start_index = max(0, min(start_index, total_rows - 1))
            end_index = max(start_index + 1, min(end_index, total_rows))

            logger.info(f"Using examples from index {start_index} to {end_index - 1} (total: {end_index - start_index})")

            df_slice = df.iloc[start_index:end_index]

        return df_slice

    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise


def prepare_examples(
        df: pd.DataFrame,
        provider_names: List[str],
        rerun: bool = False,
        results_dir: str = "results",
        random_sample: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    examples = {provider: [] for provider in provider_names}

    for provider in provider_names:
        if (not rerun) or (random_sample is not None and random_sample > 0) or (not os.path.exists(f"{results_dir}/{provider}_results.csv")):
            for _, row in df.iterrows():
                examples[provider].append({
                    "question": row["problem"],
                    "answer": row["answer"],
                    "index": int(row["index"])
                })
        else:
            if os.path.exists(f"{results_dir}/{provider}_results.csv"):
                df_results = pd.read_csv(f"{results_dir}/{provider}_results.csv")
                processed_indices = df_results[df_results['grade'] != 'ERROR']['index'].tolist()
            else:
                processed_indices = []

            # Remove rows with indices in processed_indices
            provider_df = df[~df['index'].isin(processed_indices)]
            logger.info(f"[{provider}] Removed {len(processed_indices)} already processed examples")

            for _, row in provider_df.iterrows():
                examples[provider].append({
                    "question": row["problem"],
                    "answer": row["answer"],
                    "index": int(row["index"])
                })

            logger.info(f"[{provider}] Loaded {len(examples[provider])} examples")

    return examples


def get_output_dir(output_dir: str, rerun: bool = False):
    """Get the output directory."""
    if not rerun:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(output_dir, timestamp)
    return output_dir


def save_result(result: Dict, provider_name: str, output_dir: str):
    """Appending a single result to the results CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = ['index', 'question', 'reference_answer', 'predicted_answer', 'is_correct', 'grade']

    file_exists = os.path.exists(f"{output_dir}/{provider_name}_results.csv")
    write_mode = 'a' if file_exists else 'w'
    output_file = f"{output_dir}/{provider_name}_results.csv"

    with open(output_file, write_mode) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
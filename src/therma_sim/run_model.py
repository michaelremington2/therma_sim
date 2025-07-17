import argparse
import time
import os
import sys
import yaml
import json
import logging
from model import ThermaSim


def setup_logger(output_folder):
    logger = logging.getLogger("thermasim")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Ensure we don't add multiple handlers on rerun
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(os.path.join(output_folder, "run.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run ThermaSim model with a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML or JSON configuration file.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="", help="Directory to save output files.")
    parser.add_argument("--sim_id", type=str, default="", help="ID label for simulation.")
    parser.add_argument("--snake_sample_frequency", type=int, default=0, help="Frequency of snake sampling (0 for no sampling).")
    args = parser.parse_args()

    # Load configuration
    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif args.config.endswith(".json"):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml")

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Set up logger
    logger = setup_logger(args.output)
    logger.info("=== Starting Simulation ===")
    logger.info(f"Seed: {args.seed} | Sim ID: {args.sim_id} | Output Dir: {args.output}")

    try:
        start_time = time.time()
        model = ThermaSim(config=config, seed=args.seed, output_folder=args.output, sim_id=args.sim_id,snake_sample_frequency=args.snake_sample_frequency)
        model.run_model(step_count=args.steps)
        run_time = time.time() - start_time
        logger.info(f"Simulation completed successfully in {run_time:.2f} seconds.")

    except Exception as e:
        logger.exception("Simulation failed with an unexpected error.")
        raise  # Reraise for traceback in job scheduler logs

if __name__ == "__main__":
    main()


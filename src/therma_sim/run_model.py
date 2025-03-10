import json
import argparse
import time
import os
from model import ThermaSim
import agents

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Run ThermaSim model with a JSON configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run the model (optional).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="", help="Directory to save output files.")
    parser.add_argument("--sim_id", type=str, default="", help="ID label for simulation.")

    args = parser.parse_args()

    # Load the JSON configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Run the model
    start_time = time.time()
    model = ThermaSim(config=config, seed=args.seed, output_folder=args.output, sim_id=args.sim_id)
    model.run_model()

    # # Collect results
    # model_data = model.datacollector.get_model_vars_dataframe()
    # rattlesnake_data = model.datacollector.get_agenttype_vars_dataframe(agents.Rattlesnake)

    # # Save outputs to the designated folder
    # model_data.to_csv(os.path.join(args.output, "model_data.csv"), index=False)
    # rattlesnake_data.to_csv(os.path.join(args.output, "agent_data.csv"), index=False)

    run_time = time.time() - start_time
    print(f"Model run completed in {run_time:.2f} seconds. Results saved to {args.output}")

if __name__ == "__main__":
    main()

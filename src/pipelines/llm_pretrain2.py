"""
Backwards-compatible wrapper.

Prefer running:
  python -m src.cli.pretrain_llm --config_filepath configs/llm_pretrain.yaml
"""

from src.cli.pretrain_llm import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Continued Pretraining on EHR Data")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file")
    args = parser.parse_args()
    main(args.config_filepath)
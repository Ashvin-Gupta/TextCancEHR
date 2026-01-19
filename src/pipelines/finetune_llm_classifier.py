"""
Backwards-compatible wrapper.

Prefer running:
  python -m src.cli.finetune_classifier --config_filepath configs/llm_classify_no_pretrain.yaml
"""

from src.cli.finetune_classifier import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Binary Classification Fine-tuning")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file")
    args = parser.parse_args()
    main(args.config_filepath)



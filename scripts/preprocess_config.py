#!/usr/bin/env python3
"""Config preprocessor for CIRISProxy.

Reads environment variables and applies them to litellm_config.yaml.
This allows Ansible to manage configuration via env vars.

Supported env vars:
- OPENROUTER_IGNORE_PROVIDERS: Comma-separated list of providers to ignore
  Example: "Friendli,Google" or "Friendli, Google"
"""

import os
import sys
import yaml


def preprocess_config(input_path: str, output_path: str) -> None:
    """Read config, apply env var overrides, and write to output path."""
    with open(input_path, "r") as f:
        config = yaml.safe_load(f)

    # Get ignore providers from env var (comma-separated)
    ignore_providers_str = os.environ.get("OPENROUTER_IGNORE_PROVIDERS", "")
    if ignore_providers_str:
        ignore_providers = [p.strip() for p in ignore_providers_str.split(",") if p.strip()]
        print(f"[CIRISProxy] OpenRouter ignore providers: {ignore_providers}")

        # Apply to all OpenRouter models in model_list
        for model in config.get("model_list", []):
            litellm_params = model.get("litellm_params", {})
            model_name = litellm_params.get("model", "")

            if model_name.startswith("openrouter/"):
                # Initialize extra_body.provider.ignore if needed
                if "extra_body" not in litellm_params:
                    litellm_params["extra_body"] = {}
                if "provider" not in litellm_params["extra_body"]:
                    litellm_params["extra_body"]["provider"] = {}

                litellm_params["extra_body"]["provider"]["ignore"] = ignore_providers
                print(f"[CIRISProxy] Applied ignore list to: {model.get('model_name')}")

    # Write processed config
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[CIRISProxy] Config preprocessed: {input_path} -> {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "/app/config.yaml"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/app/config.processed.yaml"
    preprocess_config(input_file, output_file)

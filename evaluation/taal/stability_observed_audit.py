"""Run the ordinary six-condition audit with additional residual-rounding observations."""

from contextlib import ExitStack
from pathlib import Path
import sys
import os
import yaml
from unittest.mock import patch

from evaluation.taal import state_audit
from evaluation.taal.stability_interventions import intervene
from evaluation.taal.stability_observation import ReadRoundingRecorder


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    original = state_audit.load_model
    settings = yaml.safe_load(
        Path(sys.argv[sys.argv.index("--config") + 1]).read_text()
    )
    requested = os.environ.get("TAAL_STABILITY_INTERVENTION", "baseline")
    if settings.get("stability_intervention", "baseline") != requested:
        raise ValueError("Intervention must be declared in the saved audit settings")

    with ExitStack() as stack:

        def load(checkpoint):
            model = original(checkpoint)
            arm = requested
            stack.enter_context(intervene(model.eval(), arm))
            stack.enter_context(
                ReadRoundingRecorder(model, output / "read-rounding.jsonl")
            )
            return model

        stack.enter_context(patch.object(state_audit, "load_model", load))
        state_audit.main()


if __name__ == "__main__":
    main()

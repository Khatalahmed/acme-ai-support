"""The public /insights page shows data/insights/measured.json - it must match the result files.

build_insights.py computes it from committed evaluation results; if a result file or the builder
changes and the JSON isn't rebuilt, this fails - so the page can't show numbers no run produced.
"""

import json
from pathlib import Path

from evals import build_insights

MEASURED = Path(build_insights.OUT)


def test_measured_json_is_up_to_date():
    committed = json.loads(MEASURED.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_insights.build())), \
        "run: uv run python src/evals/build_insights.py"


def test_every_fix_names_its_source_and_improved():
    for f in build_insights.build()["fixes"]:
        assert f["source"], f["title"]
        better = f["after"] < f["before"] if f["better"] == "lower" else f["after"] > f["before"]
        assert better, f"{f['title']}: {f['before']} -> {f['after']} is not an improvement"

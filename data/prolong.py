import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from streaming.base.format.mds.reader import MDSReader


def stream_prolong(
    subset: str = "book-65536",
    revision: str | None = None,
    dataset_id: str = "princeton-nlp/prolong-data-64K",
) -> Iterator[dict[str, Any]]:
    """Yield original records, downloading each shard only when reached."""
    # Resolve once so the index and shards come from the same snapshot.
    revision = HfApi().dataset_info(dataset_id, revision=revision).sha
    index_path = Path(hf_hub_download(
        dataset_id, f"{subset}/index.json", repo_type="dataset", revision=revision,
    ))
    index = json.loads(index_path.read_text())
    for shard in index["shards"]:
        if shard["compression"] is not None:
            raise ValueError("This reader currently supports uncompressed MDS shards only")
        reader = MDSReader.from_json(dirname=str(index_path.parent), split="", obj=shard)
        reader.validate(allow_unsafe_types=False)
        hf_hub_download(
            dataset_id, f"{subset}/{shard['raw_data']['basename']}",
            repo_type="dataset", revision=revision,
        )
        for sample_index in range(shard["samples"]):
            yield reader[sample_index]

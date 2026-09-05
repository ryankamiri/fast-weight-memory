from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from scripts import prepare_prolong as prepare


def fake_row(index):
    return {
        "input_ids": [index, index], "indices": [[0, 2]], "length": 2,
        "domain": "books", "source_record_index": index, "source_length": 3,
        "length_before_truncation": 3,
    }


class PreparationTests(unittest.TestCase):
    def test_worker_initializes_tokenizers_once(self):
        with (
            patch.object(prepare, "_tokenizers", None),
            patch.object(prepare, "_tokenizer_error", None),
            patch.object(prepare.signal, "signal"),
            patch.dict(prepare.os.environ),
            patch.object(prepare.AutoTokenizer, "from_pretrained") as load,
            patch.object(prepare, "convert_record") as convert,
        ):
            prepare.initialize_worker("source-sha", "target-sha")
            self.assertEqual(load.call_count, 2)
            self.assertEqual(load.call_args_list[0].kwargs["revision"], "source-sha")
            self.assertEqual(load.call_args_list[1].kwargs["revision"], "target-sha")
            prepare.convert_task({"record": 1}, 0)
            prepare.convert_task({"record": 2}, 1)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(convert.call_count, 2)

    def test_worker_initialization_failure_reaches_task(self):
        with (
            patch.object(prepare, "_tokenizers", None),
            patch.object(prepare, "_tokenizer_error", None),
            patch.object(prepare.signal, "signal"),
            patch.dict(prepare.os.environ),
            patch.object(prepare.AutoTokenizer, "from_pretrained", side_effect=OSError("offline")),
        ):
            prepare.initialize_worker("source-sha", "target-sha")
            with self.assertRaisesRegex(RuntimeError, "offline"):
                prepare.convert_task({}, 0)

    def test_interrupt_resume_and_upload_retry(self):
        with (
            TemporaryDirectory() as temporary,
            patch.object(prepare, "OUTPUT_DIR", Path(temporary)),
            patch.object(prepare, "RECORDS_PER_SHARD", 2),
            patch.object(prepare, "HfApi") as api,
            patch.object(prepare, "stream_prolong") as stream,
            patch.object(prepare, "parallel_rows") as rows,
            patch("sys.argv", ["prepare_prolong"]),
        ):
            api.return_value.dataset_info.return_value.sha = "dataset-pinned"
            api.return_value.model_info.return_value.sha = "tokenizer-pinned"

            def interrupted(*args):
                for index in range(3):
                    yield fake_row(index)
                raise KeyboardInterrupt

            rows.side_effect = interrupted
            with self.assertRaises(KeyboardInterrupt):
                prepare.main()
            directory = Path(temporary)
            manifest = prepare.load_checkpoint(directory)
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(manifest["tokens"], 4)
            self.assertEqual(manifest["status"], "incomplete")
            first_shard = (directory / "train-00000.parquet").read_bytes()
            api.return_value.upload_folder.assert_not_called()

            rows.side_effect = lambda *args: (fake_row(i) for i in range(args[1], 5))
            api.return_value.upload_folder.side_effect = ConnectionError("offline")
            with self.assertRaises(ConnectionError):
                prepare.main()
            self.assertEqual(stream.call_args.kwargs["start_record"], 2)
            self.assertEqual(stream.call_args.kwargs["revision"], "dataset-pinned")
            self.assertEqual(rows.call_args.args[2:], ("tokenizer-pinned", "tokenizer-pinned"))
            self.assertEqual(first_shard, (directory / "train-00000.parquet").read_bytes())
            manifest = prepare.load_checkpoint(directory)
            self.assertEqual(manifest["records"], 5)
            self.assertEqual(manifest["tokens"], 10)
            self.assertEqual(manifest["dropped_tokens"], 5)
            self.assertEqual(manifest["truncated_records"], 5)
            self.assertEqual(manifest["status"], "complete")
            indices = []
            for shard in manifest["shards"]:
                indices.extend(pq.read_table(directory / shard["file"])["source_record_index"].to_pylist())
            self.assertEqual(indices, list(range(5)))

            rows.reset_mock()
            api.return_value.upload_folder.side_effect = None
            prepare.main()
            rows.assert_not_called()
            api.return_value.dataset_info.assert_called_once()
            self.assertEqual(api.return_value.model_info.call_count, 2)
            (directory / "train-00000.parquet").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare.load_checkpoint(directory)

    def test_parallel_queue_is_bounded_and_ordered(self):
        submitted = []
        outstanding = 0
        maximum = 0

        class Result:
            def __init__(self, index):
                self.index = index

            def get(self):
                nonlocal outstanding
                outstanding -= 1
                return self.index

        def submit(function, args):
            nonlocal outstanding, maximum
            submitted.append(args[1])
            outstanding += 1
            maximum = max(maximum, outstanding)
            return Result(args[1])

        with patch.object(prepare.mp, "get_context") as context:
            pool = context.return_value.Pool.return_value.__enter__.return_value
            pool.apply_async.side_effect = submit
            result = list(prepare.parallel_rows(range(20), 5, "source", "target"))
            self.assertEqual(result, list(range(5, 25)))
            self.assertLessEqual(maximum, prepare.NUM_WORKERS * 2)
            self.assertEqual(submitted, result)
            context.return_value.Pool.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()

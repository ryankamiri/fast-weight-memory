import json
import unittest
from unittest.mock import MagicMock, patch

from data.prolong import stream_prolong


class ProLongTests(unittest.TestCase):
    def test_resume_skips_completed_shards_and_rows(self):
        shards = [
            {"compression": None, "raw_data": {"basename": f"shard.{i}.mds"}, "samples": 2}
            for i in range(2)
        ]
        reader = MagicMock()
        reader.__getitem__.side_effect = lambda j: {"id": 2 + j}
        with (
            patch("data.prolong.HfApi"),
            patch("data.prolong.hf_hub_download", return_value="/cache/books/index.json") as download,
            patch("data.prolong.Path.read_text", return_value=json.dumps({"shards": shards})),
            patch("data.prolong.MDSReader.from_json", return_value=reader),
        ):
            self.assertEqual(list(stream_prolong(start_record=3)), [{"id": 3}])
            self.assertEqual(download.call_count, 2)
            self.assertTrue(download.call_args.args[1].endswith("shard.1.mds"))
            reader.__getitem__.assert_called_once_with(1)

    def test_custom_dataset_id(self):
        shard = {"compression": None, "raw_data": {"basename": "shard.mds"}, "samples": 1}
        with (
            patch("data.prolong.HfApi") as api,
            patch("data.prolong.hf_hub_download", return_value="/cache/books/index.json") as download,
            patch("data.prolong.Path.read_text", return_value=json.dumps({"shards": [shard]})),
            patch("data.prolong.MDSReader.from_json"),
        ):
            api.return_value.dataset_info.return_value.sha = "resolved-revision"
            next(stream_prolong(subset="books", revision="v1", dataset_id="example/other-mds"))
            api.return_value.dataset_info.assert_called_once_with("example/other-mds", revision="v1")
            self.assertEqual(download.call_count, 2)
            for call in download.call_args_list:
                self.assertEqual(call.args[0], "example/other-mds")

    def test_lazy_download_and_shard_order(self):
        shards = [
            {"compression": None, "raw_data": {"basename": f"shard.{i}.mds"}, "samples": 2}
            for i in range(2)
        ]
        readers = [MagicMock(), MagicMock()]
        for i, reader in enumerate(readers):
            reader.__getitem__.side_effect = lambda j, i=i: {"id": i * 2 + j}
        with (
            patch("data.prolong.HfApi") as api,
            patch("data.prolong.hf_hub_download", return_value="/cache/book-65536/index.json") as download,
            patch("data.prolong.Path.read_text", return_value=json.dumps({"shards": shards})),
            patch("data.prolong.MDSReader.from_json", side_effect=readers),
        ):
            api.return_value.dataset_info.return_value.sha = "resolved-revision"
            stream = stream_prolong()
            download.assert_not_called()
            self.assertEqual(next(stream), {"id": 0})
            self.assertEqual(download.call_count, 2)  # Index and first shard only.
            self.assertEqual(next(stream), {"id": 1})
            self.assertEqual(download.call_count, 2)
            self.assertEqual(next(stream), {"id": 2})
            self.assertEqual(download.call_count, 3)
            self.assertEqual(list(stream), [{"id": 3}])
            for call in download.call_args_list:
                self.assertEqual(call.kwargs["revision"], "resolved-revision")
            for reader in readers:
                reader.validate.assert_called_once_with(allow_unsafe_types=False)


if __name__ == "__main__":
    unittest.main()

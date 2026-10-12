"""Offline checks of SIFT headers, range refusal and retained download evidence."""

import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "download_sift", repo / "benchmark/IVF/download_sift.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "output.bin"
        # Small synthetic responses; no real vectors, queries or network access.
        self.rows = 3
        self.base_header = struct.pack("<II", download.SOURCE_ROWS, download.DIM)
        self.base_data = bytes(self.rows * download.DIM)
        self.range = (
            f"bytes 0-{8 + len(self.base_data) - 1}/{8 + download.SOURCE_ROWS * download.DIM}"
        )

    def response(self, data=None, status=206, content_range=None):
        return Response(
            self.base_header + self.base_data if data is None else data,
            status,
            self.range if content_range is None else content_range,
        )

    def test_base_prefix(self):
        with patch.object(
            download.urllib.request, "urlopen", return_value=self.response()
        ) as request:
            download.download_base(self.path, self.rows)
        self.assertEqual(
            self.path.read_bytes(), struct.pack("<II", self.rows, download.DIM) + self.base_data
        )
        sent = request.call_args.args[0]
        self.assertEqual(sent.get_header("Range"), "bytes=0-391")
        self.assertEqual(sent.get_header("Accept-encoding"), "identity")

    def test_bad_base_responses(self):
        for response in (
            self.response(status=200),
            self.response(content_range="wrong range"),
            self.response(data=b"short"),
            self.response(data=struct.pack("<II", 1, download.DIM) + self.base_data),
        ):
            with patch.object(download.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download.download_base(self.path, self.rows)
            self.assertFalse(self.path.exists())

    def test_incomplete_base_retained(self):
        with patch.object(
            download.urllib.request,
            "urlopen",
            return_value=self.response(data=self.base_header + b"x"),
        ):
            with self.assertRaisesRegex(ValueError, "partial file retained"):
                download.download_base(self.path, self.rows)
        self.assertEqual(self.path.read_bytes(), struct.pack("<II", self.rows, download.DIM) + b"x")

    def test_queries(self):
        valid = struct.pack("<II", download.QUERY_ROWS, download.DIM) + bytes(
            download.QUERY_ROWS * download.DIM
        )
        for data in (b"", valid[:-1], valid + b"x", b"bad head" + valid[8:]):
            with patch.object(download.urllib.request, "urlopen", return_value=Response(data)):
                with self.assertRaises(ValueError):
                    download.download_queries(self.path)
            self.assertFalse(self.path.exists())
        with patch.object(download.urllib.request, "urlopen", return_value=Response(valid)):
            download.download_queries(self.path)
        self.assertEqual(self.path.read_bytes(), valid)

    def test_existing_outputs(self):
        self.path.write_bytes(b"retained evidence")
        with patch.object(download.urllib.request, "urlopen") as request:
            for call in (
                lambda: download.download_base(self.path, self.rows),
                lambda: download.download_queries(self.path),
            ):
                with self.assertRaises(FileExistsError):
                    call()
            request.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"retained evidence")

    def test_cli_preflight(self):
        self.path.write_bytes(b"retained queries")
        base = self.path.parent / "base.bin"
        argv = [
            "download_sift.py",
            "--rows",
            "10000000",
            "--base",
            str(base),
            "--query",
            str(self.path),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            download.urllib.request, "urlopen"
        ) as request:
            with self.assertRaises(SystemExit) as refused:
                download.main()
            self.assertEqual(refused.exception.code, 1)
            request.assert_not_called()
        self.assertFalse(base.exists())
        self.assertEqual(self.path.read_bytes(), b"retained queries")

    def test_cli_same_output(self):
        argv = [
            "download_sift.py",
            "--rows",
            "10000000",
            "--base",
            str(self.path),
            "--query",
            str(self.path),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            download.urllib.request, "urlopen"
        ) as request:
            with self.assertRaises(SystemExit) as refused:
                download.main()
            self.assertEqual(refused.exception.code, 1)
            request.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_dangling_symlink(self):
        try:
            self.path.symlink_to(self.path.parent / "missing.bin")
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with patch.object(download.urllib.request, "urlopen") as request:
            with self.assertRaises(FileExistsError):
                download.download_queries(self.path)
            request.assert_not_called()
        self.assertTrue(self.path.is_symlink())

    def test_cli_profiles(self):
        for rows in (10_000_000, 100_000_000):
            base, query = self.path.parent / "base.bin", self.path.parent / "query.bin"
            argv = [
                "download_sift.py",
                "--rows",
                str(rows),
                "--base",
                str(base),
                "--query",
                str(query),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                download, "download_base"
            ) as base_download, patch.object(download, "download_queries") as query_download:
                download.main()
                base_download.assert_called_once_with(base, rows)
                query_download.assert_called_once_with(query)

    def test_invalid_rows(self):
        argv = [
            "download_sift.py",
            "--rows",
            "3",
            "--base",
            str(self.path),
            "--query",
            str(self.path.parent / "query.bin"),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            download.urllib.request, "urlopen"
        ) as request:
            with self.assertRaises(SystemExit) as refused:
                download.main()
            self.assertEqual(refused.exception.code, 2)
            request.assert_not_called()
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()

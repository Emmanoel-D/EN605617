#!/usr/bin/env python3
"""Integration checks for real OpenCL execution and emitted reports."""

import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
SUFFIX = ".exe" if os.name == "nt" else ""
BINARY = PROJECT / "build" / ("motor_monitor" + SUFFIX)
COUNT_FIELDS = ("temperature_flags", "current_flags",
                "vibration_flags", "speed_flags")
FAULTS = ("overheating", "overcurrent", "vibration", "speed_deviation")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class IntegrationTests(unittest.TestCase):
    def invoke(self, args, success=True):
        result = subprocess.run(
            [str(BINARY), *args], cwd=PROJECT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def execute(self, directory, *args):
        self.invoke(["--iterations", "1", "--export-samples", "0",
                     "--output", str(directory), *args])
        rows = csv_rows(directory / "summary.csv")
        self.assertEqual([r["mode"] for r in rows], ["naive", "tiled"])
        for row in rows:
            self.assertEqual(row["validation"], "PASS")
        for key in COUNT_FIELDS:
            self.assertEqual(rows[0][key], rows[1][key])
        return rows

    def test_help_needs_no_device(self):
        self.assertIn("electric motor", self.invoke(["--help"]))

    def test_rejects_invalid_arguments(self):
        cases = (["--window", "4"], ["--samples", "0"],
                 ["--samples", "-1"], ["--window", "131"],
                 ["--local-size", "96"], ["--mode", "unknown"],
                 ["--temperature-limit", "nan"],
                 ["--vibration-limit", "inf"],
                 ["--samples", "22x"], ["--samples"], ["--bogus", "3"])
        for args in cases:
            with self.subTest(args=args):
                self.assertIn("ERROR:", self.invoke(args, success=False))

    def test_clamped_endpoints_and_partial_groups(self):
        cases = ((1, 129, 32), (17, 3, 64), (127, 33, 128),
                 (128, 33, 128), (129, 129, 32), (257, 129, 256))
        with tempfile.TemporaryDirectory() as temp:
            for n, window, local in cases:
                with self.subTest(samples=n, window=window, local=local):
                    self.execute(Path(temp), "--samples", str(n),
                                 "--window", str(window),
                                 "--local-size", str(local))

    def test_faults_reports_and_chart(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = self.execute(directory, "--samples", "8193",
                                "--export-samples", "8193")
            self.assertTrue(all(int(rows[0][k]) > 0 for k in COUNT_FIELDS))
            for mode in ("naive", "tiled"):
                folder = directory / mode
                events = csv_rows(folder / "events.csv")
                for fault, key in zip(FAULTS, COUNT_FIELDS):
                    selected = [r for r in events if r["fault"] == fault]
                    duration = sum(int(r["end_sample_exclusive"]) -
                                   int(r["start_sample"]) for r in selected)
                    self.assertEqual(duration, int(rows[0][key]))
                for row in csv_rows(folder / "fault_metrics.csv"):
                    self.assertGreater(float(row["precision"]), 0.90)
                    self.assertGreater(float(row["recall"]), 0.90)
                self.assertEqual(len(csv_rows(folder / "trace.csv")), 8193)
                ET.parse(folder / "telemetry.svg")

    def test_custom_limits_disable_all_faults(self):
        args = ["--samples", "8192"]
        for channel in ("temperature", "current", "vibration", "speed"):
            args.extend([f"--{channel}-limit", "100000"])
        with tempfile.TemporaryDirectory() as temp:
            rows = self.execute(Path(temp), *args)
            for row in rows:
                self.assertTrue(all(int(row[k]) == 0 for k in COUNT_FIELDS))

    def test_out_of_order_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            args = ["--samples", "16385", "--window", "129",
                    "--local-size", "64", "--seed", "4294967295",
                    "--queue", "out-of-order"]
            try:
                self.execute(Path(temp), *args)
            except AssertionError as error:
                if "Device lacks out-of-order queues" in str(error):
                    self.skipTest("Device lacks out-of-order queues")
                raise


if __name__ == "__main__":
    unittest.main(verbosity=2)

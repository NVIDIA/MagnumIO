# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tests.container_matrix.test_container_matrix import _extract_highlights, _run, _write_runner


class ContainerMatrixReportTests(unittest.TestCase):
    def test_generated_runner_propagates_inner_command_exit(self):
        with TemporaryDirectory() as tmp:
            runner = Path(tmp) / "runner.py"
            _write_runner(runner)

            content = runner.read_text(encoding="utf-8")

        self.assertIn("overall_rc = 0", content)
        self.assertIn("overall_rc = max(overall_rc, completed.returncode)", content)
        self.assertIn("sys.exit(overall_rc)", content)

    def test_run_records_timeout_as_case_failure(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "case.log"
            timeout = subprocess.TimeoutExpired(
                cmd=["sleep", "99"],
                timeout=3,
                output="partial output",
                stderr="partial stderr",
            )
            with mock.patch(
                "tests.container_matrix.test_container_matrix.subprocess.run",
                side_effect=timeout,
            ):
                rc = _run(log, ["sleep", "99"], 3)

            content = log.read_text(encoding="utf-8")

        self.assertEqual(rc, 124)
        self.assertIn("[TIMEOUT] command exceeded 3 seconds", content)
        self.assertIn("TimeoutExpired", content)
        self.assertIn("partial output", content)
        self.assertIn("partial stderr", content)
        self.assertIn("[case-exit:124]", content)

    def test_extract_highlights_omits_verbose_config_values(self):
        text = """
$ python3 gds-diag.py config-audit -v
  Config source : gdscheck -p CUFILE CONFIGURATION
  File fallback : /etc/cufile.json
      value  : 'ERROR'  (file fallback)
      value  : True  (gdscheck)
  WARN  properties.max_device_cache_size_kb/properties.per_buffer_cache_size_kb/properties.io_batchsize
      Device cache sizing may limit batching.
      value  : {'max_device_cache_size_kb': 393216, 'per_buffer_cache_size_kb': 16384, 'io_batchsize': 128}  (derived)
      risk   : Device bounce-buffer cache cannot satisfy the configured cuFile batch size
      action : Increase max_device_cache_size_kb, lower per_buffer_cache_size_kb, or lower properties.io_batchsize.
[exit:0] python3 gds-diag.py config-audit -v
"""

        highlights = "\n".join(_extract_highlights(text))

        self.assertIn("$ python3 gds-diag.py config-audit -v", highlights)
        self.assertIn("WARN  properties.max_device_cache_size_kb", highlights)
        self.assertIn("[exit:0] python3 gds-diag.py config-audit -v", highlights)
        self.assertNotIn("value  : 'ERROR'", highlights)
        self.assertNotIn("Device bounce-buffer cache cannot satisfy", highlights)
        self.assertNotIn("Increase max_device_cache_size_kb", highlights)
        self.assertNotIn("File fallback", highlights)

    def test_extract_highlights_keeps_wrapped_warn_table_rows(self):
        text = """
$ python3 gds-diag.py container-check -v
  | udev database                | WARN   | /run/udev is missing or empty. GDS may be unable | Add `-m                                                |
  |                              |        | to resolve block-device metadata correctly from  | /run/udev:/run/udev:none:x-create=dir,rbind,ro:0:0` to |
  |                              |        | inside this container.                           | the Enroot start command.                              |
  14 passed, 2 info, 2 warnings
  Container Launch Recommendations
  1. WARN udev database
     Action:
       Add `-m /run/udev:/run/udev:none:x-create=dir,rbind,ro:0:0` to the Enroot start command.
[exit:0] python3 gds-diag.py container-check -v
"""

        highlights = "\n".join(_extract_highlights(text))

        self.assertIn("udev database", highlights)
        self.assertIn("/run/udev:/run/udev:none:x-create=dir,rbind,ro:0:0", highlights)
        self.assertIn("14 passed, 2 info, 2 warnings", highlights)


if __name__ == "__main__":
    unittest.main()

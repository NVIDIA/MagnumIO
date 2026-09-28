# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Result types shared across all GDS pre-checks.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Status(str, Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"
    NA   = "N/A"   # mode not applicable for this filesystem type


class GDSMode(str, Enum):
    NATIVE   = "Native GDS (nvidia-fs)"
    P2PDMA   = "P2P DMA/C2C (NVMe↔GPU direct)"
    RDMA     = "RDMA (network storage)"
    COMPAT   = "Compat / CPU bounce buffer"


@dataclass
class CheckResult:
    """A single sub-check contributing to a mode's overall verdict."""
    check: str                      # human label, e.g. "IOMMU mode"
    mode: GDSMode
    status: Status
    why: str                        # technical root-cause sentence
    mitigation: Optional[str] = None
    evidence: Optional[str] = None  # raw snippet that led to the verdict
    # True when this FAIL is downgraded from blocking because a confirmed
    # working alternate direct GDS path (P2PDMA vs. nvidia-fs/nvfs) makes it
    # non-essential. Set by gds_report._mark_alternate_path_exemptions().
    exempted: bool = False


@dataclass
class ModeReport:
    """Aggregated result for one GDS mode."""
    mode: GDSMode
    applicable: bool                # False = FS does not support this mode at all
    results: List[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if not self.applicable:
            return Status.NA
        if not self.results:
            return Status.PASS
        if any(r.status == Status.FAIL and not r.exempted for r in self.results):
            return Status.FAIL
        if any(r.status == Status.WARN or (r.status == Status.FAIL and r.exempted) for r in self.results):
            return Status.WARN
        if any(r.status == Status.INFO for r in self.results):
            return Status.INFO
        return Status.PASS

    def blockers(self) -> List[CheckResult]:
        """Hard, non-exempted failures — used for the summary table Blockers column."""
        return [r for r in self.results if r.status == Status.FAIL and not r.exempted]

    def failing(self) -> List[CheckResult]:
        """FAILs and WARNs — used for detailed findings section."""
        return [r for r in self.results if r.status in (Status.FAIL, Status.WARN)]

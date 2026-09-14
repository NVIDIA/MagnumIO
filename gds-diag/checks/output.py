# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared ANSI formatting and rendering utilities for gds-diag subcommands.
"""
from __future__ import annotations

import sys
import textwrap
import re

from .result import CheckResult, Status

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def green(t: str) -> str:  return _c("32", t)
def cyan(t: str) -> str:   return _c("36", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str:    return _c("31", t)
def bold(t: str) -> str:   return _c("1",  t)
def dim(t: str) -> str:    return _c("2",  t)


STATUS_ICON = {
    Status.PASS: green("✓ PASS"),
    Status.INFO: cyan("ℹ INFO"),  # noqa: RUF001
    Status.WARN: yellow("⚠ WARN"),
    Status.FAIL: red("✗ FAIL"),
    Status.NA:   dim("– N/A "),
}


URL_RE = re.compile(r"https?://[^\s<>()|]+")


def _fold_statuses(statuses: list[Status]) -> Status:
    """Aggregate a list of statuses down to one severity, worst first."""
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.WARN in statuses:
        return Status.WARN
    if Status.INFO in statuses:
        return Status.INFO
    return Status.PASS


def section_status(results: list[CheckResult]) -> Status:
    """Aggregate status for a group of results."""
    return _fold_statuses([r.status for r in results])


def render_section(
    title: str,
    results: list[CheckResult],
    verbose: bool = False,
    *,
    status: "Status | None" = None,
) -> list[str]:
    """Render a named group of CheckResults as a ┌─│└─ box.

    ``status`` overrides the computed section status — use this when the
    caller already knows the aggregate status (e.g. ModeReport.status).
    """
    lines: list[str] = []
    if status is None:
        status = section_status(results)

    lines.append(bold(f"  ┌─ {title}") + f"  {STATUS_ICON[status]}")

    if status == Status.PASS and not verbose:
        lines.append("  │  All checks passed.")
    else:
        to_show = results if verbose else [
            r for r in results if r.status not in (Status.PASS,)
        ]
        for result in to_show:
            icon = STATUS_ICON[_display_status(result)]
            suffix = dim("  (optional — alternate direct path active)") if result.exempted else ""
            lines.append("  │")
            lines.append(f"  │  {icon}  {bold(result.check)}{suffix}")
            lines.append(f"  │  {textwrap.fill(result.why, width=62, subsequent_indent='  │  ')}")
            if result.evidence and verbose:
                lines.append(f"  │  {dim('Evidence:')}")
                for ev_line in result.evidence.splitlines()[:4]:
                    lines.append(f"  │    {dim(ev_line)}")
            if result.mitigation:
                lines.append("  │")
                lines.append(f"  │  {yellow('→ Mitigation:')}")
                lines.extend(_format_block_text(result.mitigation, indent="  │    ", width=80))

    lines.append(f"  └{'─' * 60}")
    lines.append("")
    return lines


def _format_block_text(text: str, *, indent: str = "      ", width: int = 80) -> list[str]:
    """Format free text for block output while keeping URLs intact.

    ``width`` is the total line width including the indent prefix.
    """
    wrap_width = max(width - len(indent), 20)
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        raw_line = raw_line.rstrip()
        if not raw_line:
            lines.append("")
            continue
        # Indented command examples should remain directly copyable.
        if raw_line.startswith((" ", "\t")):
            lines.append(f"{indent}{raw_line}")
            continue
        if URL_RE.search(raw_line):
            cursor = 0
            for match in URL_RE.finditer(raw_line):
                before = raw_line[cursor:match.start()].strip()
                if before:
                    wrapped = textwrap.wrap(
                        before,
                        width=wrap_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    ) or [""]
                    lines.extend(f"{indent}{line}" for line in wrapped)
                lines.append(f"{indent}{match.group(0)}")
                cursor = match.end()
            after = raw_line[cursor:].strip()
            if after:
                wrapped = textwrap.wrap(
                    after,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [""]
                lines.extend(f"{indent}{line}" for line in wrapped)
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        lines.extend(f"{indent}{line}" for line in wrapped)
    return lines


def _format_field(label: str, text: str | None, *, indent: str = "    ") -> list[str]:
    if not text:
        return []
    lines = [f"{indent}{label}:"]
    lines.extend(_format_block_text(text, indent=indent + "  "))
    return lines


def _wrap_cell(value: str, width: int) -> list[str]:
    text = value if value else "-"
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
        )
    return lines or [""]


def _table_safe_text(text: str | None) -> str:
    if not text:
        return ""
    return URL_RE.sub("See final Mitigation Plan for link.", text)


def _table_line(widths: list[int], char: str = "-") -> str:
    body = "+".join(char * (width + 2) for width in widths)
    return f"  +{body}+"


def _plain_status(status: Status) -> str:
    return status.value


def _display_status(result: CheckResult) -> Status:
    """Status to show in badges/icons — an exempted FAIL displays as WARN,
    matching ModeReport.status's downgrade so a reader never sees a bare
    FAIL badge inside a section the aggregate status already calls WARN."""
    return Status.WARN if result.exempted else result.status


def _default_action(result: CheckResult) -> str:
    if result.exempted:
        base = result.mitigation or "No action required."
        return (
            f"{base}\n"
            "(Optional — an alternate direct GDS path is already confirmed working; "
            "this is a performance opportunity, not a blocker.)"
        )
    if result.status == Status.INFO:
        return result.mitigation or "No action required."
    if result.mitigation:
        return result.mitigation
    if result.status == Status.PASS:
        return "No action needed."
    if result.status == Status.WARN:
        return "Review before workload testing; fix if unexpected."
    if result.status == Status.FAIL:
        return "Fix this before continuing."
    return "Not applicable."


def _issue_results(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.status in (Status.FAIL, Status.WARN, Status.INFO)]


def render_section_table(
    title: str,
    results: list[CheckResult],
    verbose: bool = False,
) -> list[str]:
    """Render one section as an ASCII table.

    Normal mode shows informational notes, warnings, and failures. Verbose mode
    shows every row.
    """
    display_results = results if verbose else _issue_results(results)
    if not display_results:
        return []

    status = _fold_statuses([_display_status(r) for r in results])

    widths = [24, 6, 38]
    headers = ["Check", "Status", "Finding"]

    def render_row(values: list[str]) -> list[str]:
        values = [_table_safe_text(value) for value in values]
        wrapped = [_wrap_cell(value, width) for value, width in zip(values, widths)]
        height = max(len(cell) for cell in wrapped)
        rows: list[str] = []
        for idx in range(height):
            cells = []
            for cell, width in zip(wrapped, widths):
                value = cell[idx] if idx < len(cell) else ""
                cells.append(f" {value:<{width}} ")
            rows.append("  |" + "|".join(cells) + "|")
        return rows

    lines = [bold(f"  {title}") + f"  {STATUS_ICON[status]}"]
    lines.append(_table_line(widths, "="))
    lines.extend(render_row(headers))
    lines.append(_table_line(widths, "="))
    for result in display_results:
        finding = result.why
        if verbose and result.evidence:
            evidence = "\n".join(result.evidence.splitlines()[:4])
            finding = f"{finding} Evidence: {evidence}"
        lines.extend(render_row([
            result.check,
            _plain_status(_display_status(result)),
            finding,
        ]))
        lines.append(_table_line(widths))

    lines.append("")
    return lines


def render_sections_table(
    sections: dict[str, list[CheckResult]],
    verbose: bool = False,
) -> list[str]:
    """Render sections as ASCII tables."""
    lines: list[str] = []
    for title, results in sections.items():
        lines.extend(render_section_table(title, results, verbose=verbose))
    if not lines and not verbose:
        lines.append("  No warnings or errors.")
        lines.append("")
    return lines


def render_sections(
    sections: dict[str, list[CheckResult]],
    verbose: bool = False,
) -> list[str]:
    """Render sections using the box style.

    Normal mode skips all-pass sections. Verbose mode renders every section.
    """
    lines: list[str] = []
    for title, results in sections.items():
        if not verbose and all(r.status == Status.PASS for r in results):
            continue
        lines.extend(render_section(title, results, verbose=verbose))
    if not lines and not verbose:
        lines.append("  No warnings or errors.")
        lines.append("")
    return lines


def render_version_context(rows: list[dict[str, object]]) -> list[str]:
    """Render always-visible runtime/package version context."""
    if not rows:
        return []

    lines = [bold("  Runtime Versions")]
    label_width = max(len(str(row.get("component", ""))) for row in rows)
    for row in rows:
        component = str(row.get("component", ""))
        version = str(row.get("version") or "unknown")
        prefix = f"    {component:<{label_width}} : "
        wrapped = textwrap.wrap(
            version,
            width=max(80 - len(prefix), 20),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        lines.append(prefix + wrapped[0])
        continuation = " " * len(prefix)
        for line in wrapped[1:]:
            lines.append(continuation + line)
    lines.append("")
    return lines


def render_mitigation_plan(
    sections: dict[str, list[CheckResult]],
    *,
    title: str = "Mitigation Plan",
) -> list[str]:
    """Render a final action plan for WARN/FAIL checks."""
    issues: list[tuple[str, CheckResult]] = []
    for section, results in sections.items():
        for result in results:
            if result.status in (Status.FAIL, Status.WARN):
                issues.append((section, result))

    # Exempted (non-blocking) items are optional follow-ups, not required
    # fixes — list them after everything that's actually blocking.
    issues.sort(key=lambda pair: pair[1].exempted)

    lines = [bold(f"  {title}")]
    if not issues:
        lines.append("  No warnings or errors require mitigation.")
        lines.append("")
        return lines

    for idx, (section, result) in enumerate(issues, start=1):
        action = _default_action(result)
        label = "OPT " if result.exempted else f"{_plain_status(result.status):<4}"
        lines.append(f"  {idx}. {label} {bold(result.check)}")
        lines.append(f"     Section: {section}")
        lines.extend(_format_field("Action", action, indent="     "))

    lines.append("")
    return lines


def render_summary(sections: dict[str, list[CheckResult]]) -> str:
    """One-line pass/info/warn/fail tally across all sections."""
    all_results = [r for results in sections.values() for r in results]
    statuses = [_display_status(r) for r in all_results]
    n_pass = sum(1 for s in statuses if s == Status.PASS)
    n_info = sum(1 for s in statuses if s == Status.INFO)
    n_warn = sum(1 for s in statuses if s == Status.WARN)
    n_fail = sum(1 for s in statuses if s == Status.FAIL)
    parts = []
    if n_pass:
        parts.append(green(f"{n_pass} passed"))
    if n_info:
        parts.append(cyan(f"{n_info} info"))
    if n_warn:
        parts.append(yellow(f"{n_warn} warning{'s' if n_warn != 1 else ''}"))
    if n_fail:
        parts.append(red(f"{n_fail} failed"))
    return "  " + ", ".join(parts) if parts else "  No checks run."


def results_to_json(results: list[CheckResult]) -> list[dict]:
    return [
        {
            "check": r.check,
            "status": _display_status(r).value,
            "finding": r.why,
            "mitigation": r.mitigation,
            "evidence": r.evidence,
        }
        for r in results
    ]


def overall_exit_code(sections: dict[str, list[CheckResult]]) -> int:
    """0 = no FAIL, 1 = at least one FAIL (exempted FAILs display as WARN and
    do not count — see _display_status())."""
    all_results = [r for results in sections.values() for r in results]
    return 1 if any(_display_status(r) == Status.FAIL for r in all_results) else 0

#!/usr/bin/env python3
"""
Lightweight browser for kept/rejected ticket JSONL produced by the Ray pipeline.

Goals:
- Arrow-key TUI for one-line summaries and drill-down viewing.
- No third-party dependencies (standard library only).

Supported inputs:
- A directory produced by Ray `Dataset.write_json(...)` (many *.json files with JSONL).
- A single JSONL/JSON file.

Notes:
- Random access is supported for uncompressed files via byte offsets.
- Gzip inputs are supported for listing, but "show by index" will re-scan.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


JSONL_SUFFIXES = (".jsonl", ".json")
GZIP_SUFFIXES = (".gz",)


@dataclass(frozen=True)
class RecordRef:
    dataset: str  # "kept" or "rejected"
    path: str
    offset: int | None  # byte offset for uncompressed files
    incident_number: str
    state: str
    short_description: str
    rejection_reasons: tuple[str, ...]


def _iter_input_files(root: str) -> list[str]:
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {root}")

    if p.is_file():
        return [str(p)]

    files: list[str] = []
    for child in sorted(p.rglob("*")):
        if not child.is_file():
            continue
        name = child.name.lower()
        if name.startswith("."):
            continue
        if name.endswith(JSONL_SUFFIXES) or name.endswith(tuple(s + g for s in JSONL_SUFFIXES for g in GZIP_SUFFIXES)):
            files.append(str(child))
    return files


def _open_maybe_gzip(path: str, mode: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)  # noqa: P201


def _iter_records_from_file(path: str) -> Iterator[tuple[int | None, dict[str, Any]]]:
    """
    Yield (offset, record) for each JSON line in the file.

    If the file is gzip-compressed, offset is None (not seekable in a useful way).
    """
    if path.lower().endswith(".gz"):
        with _open_maybe_gzip(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield None, json.loads(line)
        return

    with _open_maybe_gzip(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            yield offset, obj


def _extract_one_line_fields(
    record: Mapping[str, Any],
) -> tuple[str, str, str, tuple[str, ...]]:
    incident_number = str(
        record.get("incident_number")
        or record.get("number")
        or ""
    ).strip()
    state = str(record.get("state") or "").strip()
    short_description = str(record.get("short_description") or "").strip()
    reasons_raw = record.get("rejection_reasons") or []
    if isinstance(reasons_raw, list):
        reasons = tuple(str(x) for x in reasons_raw if x)
    else:
        reasons = (str(reasons_raw),) if reasons_raw else ()
    return incident_number, state, short_description, reasons


def _index_dataset(dataset_name: str, root: str, *, limit: int | None = None) -> list[RecordRef]:
    refs: list[RecordRef] = []
    files = _iter_input_files(root)
    for path in files:
        for offset, rec in _iter_records_from_file(path):
            if not isinstance(rec, dict):
                continue
            incident_number, state, short_description, reasons = _extract_one_line_fields(rec)
            refs.append(
                RecordRef(
                    dataset=dataset_name,
                    path=path,
                    offset=offset,
                    incident_number=incident_number,
                    state=state,
                    short_description=short_description,
                    rejection_reasons=reasons,
                )
            )
            if limit is not None and len(refs) >= limit:
                return refs
    return refs


def _format_row(ref: RecordRef, idx: int, *, width: int = 120) -> str:
    tag = "K" if ref.dataset == "kept" else "R"
    incident = ref.incident_number or "-"
    state = ref.state or "-"
    short = ref.short_description.replace("\n", " ").strip() or "-"
    reasons = ""
    if ref.dataset != "kept":
        reasons = " reasons=" + ",".join(ref.rejection_reasons[:4])
        if len(ref.rejection_reasons) > 4:
            reasons += ",..."
    base = f"[{tag}] {idx:6d}  {incident:12s}  {state:10s}  {short}{reasons}"
    if len(base) <= width:
        return base
    return base[: max(0, width - 1)] + "…"


def _load_record(ref: RecordRef) -> dict[str, Any] | None:
    if ref.offset is None:
        # gzip: re-scan the file until we find a matching incident_number + short_description.
        target = (ref.incident_number, ref.short_description)
        for _, rec in _iter_records_from_file(ref.path):
            if not isinstance(rec, dict):
                continue
            inc, _, short, _ = _extract_one_line_fields(rec)  # type: ignore[arg-type]
            if (inc, short) == target:
                return rec
        return None

    with open(ref.path, "rb") as f:  # noqa: P201
        f.seek(ref.offset)
        line = f.readline()
    try:
        return json.loads(line.decode("utf-8"))
    except Exception:
        return None


def _wrap_lines(text: str, *, width: int) -> list[str]:
    if width <= 4:
        return [text]
    out: list[str] = []
    for line in (text or "").splitlines() or [""]:
        wrapped = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=True,
        )
        if not wrapped:
            out.append("")
        else:
            out.extend(wrapped)
    return out


def _record_to_detail_lines(rec: dict[str, Any], *, width: int, raw_json: bool) -> list[str]:
    if raw_json:
        blob = json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=True)
        return _wrap_lines(blob, width=width)

    lines: list[str] = []

    def add_kv(key: str) -> None:
        v = rec.get(key)
        if v is None or v == "":
            return
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
        else:
            s = str(v)
        for w in _wrap_lines(f"{key}: {s}", width=width):
            lines.append(w)

    for k in (
        "incident_number",
        "number",
        "sys_id",
        "state",
        "active",
        "opened_at",
        "closed_at",
        "priority",
        "assignment_group",
        "assigned_to",
        "sys_created_by",
        "contact_type",
        "rejection_reasons",
        "source_path",
    ):
        add_kv(k)

    lines.append("")

    for k in (
        "short_description",
        "description",
        "customer_comments_text",
        "internal_work_notes_text",
        "close_notes",
    ):
        v = rec.get(k)
        if v is None or v == "":
            continue
        lines.append(f"{k}:")
        for w in _wrap_lines(str(v), width=max(10, width - 2)):
            lines.append("  " + w)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()
    return lines


def run_tui(args: argparse.Namespace) -> int:
    # Import curses lazily so non-interactive uses don't require it.
    try:
        import curses
    except Exception as e:
        raise SystemExit(f"curses is required for tui mode: {e}")

    refs: list[RecordRef] = []
    if args.kept:
        refs.extend(_index_dataset("kept", args.kept, limit=args.limit_index))
    if args.rejected:
        refs.extend(_index_dataset("rejected", args.rejected, limit=args.limit_index))

    if not refs:
        print("No records found.")
        return 2

    query = ""
    visible: list[int] = list(range(len(refs)))
    selected_pos = 0
    top = 0
    mode: str = "all"  # all|kept|rejected

    def apply_filter() -> None:
        nonlocal visible, selected_pos, top
        q = query.strip().lower()
        out: list[int] = []
        for idx, r in enumerate(refs):
            if mode != "all" and r.dataset != mode:
                continue
            if not q:
                out.append(idx)
                continue
            if q in (r.incident_number or "").lower():
                out.append(idx)
                continue
            if q in (r.short_description or "").lower():
                out.append(idx)
                continue
            if q in ",".join(r.rejection_reasons).lower():
                out.append(idx)
                continue
        visible = out
        selected_pos = 0
        top = 0

    def clamp() -> None:
        nonlocal selected_pos, top
        if not visible:
            selected_pos = 0
            top = 0
            return
        selected_pos = max(0, min(selected_pos, len(visible) - 1))
        top = max(0, min(top, max(0, len(visible) - 1)))

    def prompt(stdscr: Any, label: str) -> str:
        curses.echo()
        curses.curs_set(1)
        h, w = stdscr.getmaxyx()
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()
        stdscr.addstr(h - 1, 0, label[: max(0, w - 1)])
        stdscr.refresh()
        try:
            s = stdscr.getstr(h - 1, min(len(label), max(0, w - 1))).decode("utf-8")
        except Exception:
            s = ""
        curses.noecho()
        curses.curs_set(0)
        return s

    def draw_list(stdscr: Any) -> None:
        nonlocal selected_pos, top
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        title = f"Tickets: {len(refs)}  Visible: {len(visible)}  Mode: {mode}"
        if query:
            title += f"  Search: {query!r}"
        stdscr.addstr(0, 0, title[: max(0, w - 1)])

        help_line = "↑/↓ move  Home/End start/end  PgUp/PgDn page  Enter details  t mode  / search  q quit"
        stdscr.addstr(h - 1, 0, help_line[: max(0, w - 1)])

        if not visible:
            stdscr.addstr(2, 0, "No matches. Press / to search again or clear.")
            stdscr.refresh()
            return

        list_h = max(0, h - 2)
        clamp()
        if selected_pos < top:
            top = selected_pos
        if selected_pos >= top + list_h:
            top = max(0, selected_pos - list_h + 1)

        end = min(len(visible), top + list_h)
        for row, pos in enumerate(range(top, end), start=1):
            idx = visible[pos]
            line = _format_row(refs[idx], idx, width=w - 1)
            if pos == selected_pos:
                with contextlib.suppress(Exception):
                    stdscr.addstr(row, 0, line, curses.A_REVERSE)
            else:
                stdscr.addstr(row, 0, line)
        stdscr.refresh()

    def run_detail(stdscr: Any, ref: RecordRef) -> None:
        nonlocal query
        raw_json = False
        rec = _load_record(ref) or {}
        scroll = 0
        j_down_armed = False

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            header = _format_row(ref, -1, width=w - 1).replace("[-1", "[sel")
            stdscr.addstr(0, 0, header[: max(0, w - 1)])
            help_line = "Esc back  ↑/↓ scroll  g/G start/end  PgUp/PgDn page  j toggle json  / search  q quit"
            stdscr.addstr(h - 1, 0, help_line[: max(0, w - 1)])

            lines = _record_to_detail_lines(rec, width=max(20, w - 1), raw_json=raw_json)
            body_h = max(0, h - 2)
            scroll = max(0, min(scroll, max(0, len(lines) - body_h)))

            for i in range(body_h):
                j = scroll + i
                if j >= len(lines):
                    break
                stdscr.addstr(1 + i, 0, lines[j][: max(0, w - 1)])

            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (27,):  # ESC
                return
            if ch in (ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if ch in (curses.KEY_UP, ord("k")):
                scroll = max(0, scroll - 1)
                continue
            if ch in (curses.KEY_DOWN, ord("l")):
                scroll = min(max(0, len(lines) - body_h), scroll + 1)
                continue
            if ch in (curses.KEY_HOME, ord("g")):
                scroll = 0
                continue
            if ch in (curses.KEY_END, ord("G")):
                scroll = max(0, len(lines) - body_h)
                continue
            if ch == curses.KEY_NPAGE:
                scroll = min(max(0, len(lines) - body_h), scroll + body_h)
                continue
            if ch == curses.KEY_PPAGE:
                scroll = max(0, scroll - body_h)
                continue
            if ch in (ord("j"), ord("J")):
                # Toggle JSON view. If user is accustomed to vi-style navigation,
                # allow "jj" to scroll down one line without losing the toggle.
                if j_down_armed:
                    scroll = min(max(0, len(lines) - body_h), scroll + 1)
                    j_down_armed = False
                    continue
                raw_json = not raw_json
                scroll = 0
                j_down_armed = True
                continue
            if ch == ord("/"):
                new_q = prompt(stdscr, "search> ")
                query = new_q
                apply_filter()
                return

    def main_curses(stdscr: Any) -> int:
        nonlocal selected_pos, top, query, mode
        curses.curs_set(0)
        stdscr.keypad(True)

        apply_filter()

        while True:
            draw_list(stdscr)
            ch = stdscr.getch()

            if ch in (ord("q"), ord("Q")):
                return 0
            if ch == ord("/"):
                query = prompt(stdscr, "search> ")
                apply_filter()
                continue
            if ch in (ord("t"), ord("T")):
                mode = "kept" if mode == "all" else "rejected" if mode == "kept" else "all"
                apply_filter()
                continue

            if not visible:
                continue

            h, _ = stdscr.getmaxyx()
            list_h = max(1, h - 2)

            if ch in (curses.KEY_UP, ord("k")):
                selected_pos = max(0, selected_pos - 1)
                continue
            if ch in (curses.KEY_DOWN, ord("j")):
                selected_pos = min(len(visible) - 1, selected_pos + 1)
                continue
            if ch in (curses.KEY_HOME, ord("g")):
                selected_pos = 0
                top = 0
                continue
            if ch in (curses.KEY_END, ord("G")):
                selected_pos = len(visible) - 1
                top = max(0, selected_pos - list_h + 1)
                continue
            if ch == curses.KEY_NPAGE:
                selected_pos = min(len(visible) - 1, selected_pos + list_h)
                continue
            if ch == curses.KEY_PPAGE:
                selected_pos = max(0, selected_pos - list_h)
                continue
            if ch in (10, 13, curses.KEY_ENTER):
                idx = visible[selected_pos]
                run_detail(stdscr, refs[idx])
                continue

    try:
        return int(curses.wrapper(main_curses))
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arrow-key TUI for browsing kept/rejected ticket outputs.",
    )
    parser.add_argument("--kept", default=None, help="Kept dataset dir/file (Ray write_json output).")
    parser.add_argument("--rejected", default=None, help="Rejected dataset dir/file (Ray write_json output).")
    parser.add_argument("--width", type=int, default=140, help="Max width for one-line rows.")
    parser.add_argument("--limit-index", type=int, default=None, help="Limit indexed records (debug).")
    args = parser.parse_args(argv)

    if not args.kept and not args.rejected:
        parser.error("Provide --kept and/or --rejected.")

    return int(run_tui(args))


if __name__ == "__main__":
    raise SystemExit(main())

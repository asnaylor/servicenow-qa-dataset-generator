#!/usr/bin/env python3.11
"""
Lightweight browser for browsing ServiceNow ticket JSONL shards.

Goals:
- Arrow-key TUI for one-line summaries and drill-down viewing.
- No third-party dependencies (standard library only).

Supported inputs:
- A directory containing JSONL/JSON (optionally gzipped) shards.
- A single JSONL/JSON file.

Notes:
- Random access is supported for uncompressed files via byte offsets.
- Gzip inputs are supported for listing, but "show by index" will re-scan.
"""

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
    path: str
    offset: int | None  # byte offset for uncompressed files
    line_no: int  # 0-based line number in file (used for gzip / fallback)
    incident_number: str
    sys_id: str
    state: str
    category: str
    subcategory: str
    u_resource: str
    short_description: str
    close_code: str
    source_path: str
    # Precomputed lowercase blob for fast substring search.
    search_blob: str


def _iter_input_files(root: str, *, recursive: bool) -> list[str]:
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {root}")

    if p.is_file():
        return [str(p)]

    files: list[str] = []
    it = p.rglob("*") if recursive else p.glob("*")
    for child in sorted(it):
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


def _iter_records_from_file(path: str) -> Iterator[tuple[int | None, int, dict[str, Any]]]:
    """
    Yield (offset, line_no, record) for each JSON line in the file.

    If the file is gzip-compressed, offset is None (not seekable in a useful way).
    """
    if path.lower().endswith(".gz"):
        with _open_maybe_gzip(path, "rt") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield None, line_no, json.loads(line)
        return

    with _open_maybe_gzip(path, "rb") as f:
        line_no = 0
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
                line_no += 1
                continue
            yield offset, line_no, obj
            line_no += 1


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_incident_fields(record: Mapping[str, Any]) -> Mapping[str, Any]:
    # convert_servicenow_tickets_to_jsonl.py emits nested incident_fields.
    # Ray pipeline outputs are already flattened.
    return _as_dict(record.get("incident_fields"))


def _get_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_dict(record.get("metadata"))


def _get_field(record: Mapping[str, Any], key: str) -> str:
    """
    Best-effort accessor across common schemas.

    - filtered JSONL: incident_fields.<key>
    - ray pipeline outputs: top-level <key>
    """
    v = record.get(key)
    if v is not None and v != "":
        return _norm(v)
    incident = _get_incident_fields(record)
    v = incident.get(key)
    if v is not None and v != "":
        return _norm(v)
    return ""


def _extract_one_line_fields(
    record: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    incident = _get_incident_fields(record)
    metadata = _get_metadata(record)

    incident_number = _norm(
        record.get("incident_number")
        or incident.get("number")
        or metadata.get("incident_number")
        or record.get("number")
    )
    sys_id = _norm(record.get("sys_id") or incident.get("sys_id") or metadata.get("sys_id"))
    state = _get_field(record, "state")
    category = _get_field(record, "category")
    subcategory = _get_field(record, "subcategory")
    u_resource = _get_field(record, "u_resource")
    short_description = _get_field(record, "short_description")
    close_code = _get_field(record, "close_code")
    source_path = _norm(record.get("source_path"))

    # Optional field from other tools; safe to keep for searching.
    rejection_reasons = record.get("rejection_reasons") or []
    if isinstance(rejection_reasons, list):
        reasons_text = ",".join(_norm(x) for x in rejection_reasons if _norm(x))
    else:
        reasons_text = _norm(rejection_reasons)

    return (
        incident_number,
        sys_id,
        state,
        category,
        subcategory,
        u_resource,
        short_description,
        close_code,
        source_path,
        reasons_text,
    )


def _build_search_blob(parts: Iterable[str]) -> str:
    return " ".join(p for p in (p.strip() for p in parts) if p).lower()


def _index_tickets(root: str, *, recursive: bool, limit: int | None = None) -> list[RecordRef]:
    refs: list[RecordRef] = []
    files = _iter_input_files(root, recursive=recursive)
    for path in files:
        for offset, line_no, rec in _iter_records_from_file(path):
            if not isinstance(rec, dict):
                continue
            (
                incident_number,
                sys_id,
                state,
                category,
                subcategory,
                u_resource,
                short_description,
                close_code,
                source_path,
                reasons_text,
            ) = _extract_one_line_fields(rec)

            search_blob = _build_search_blob(
                (
                    incident_number,
                    sys_id,
                    state,
                    category,
                    subcategory,
                    u_resource,
                    short_description,
                    close_code,
                    source_path,
                    reasons_text,
                )
            )
            refs.append(
                RecordRef(
                    path=path,
                    offset=offset,
                    line_no=line_no,
                    incident_number=incident_number,
                    sys_id=sys_id,
                    state=state,
                    category=category,
                    subcategory=subcategory,
                    u_resource=u_resource,
                    short_description=short_description,
                    close_code=close_code,
                    source_path=source_path,
                    search_blob=search_blob,
                )
            )
            if limit is not None and len(refs) >= limit:
                return refs
    return refs


def _format_row(ref: RecordRef, idx: int, *, width: int = 120) -> str:
    incident = ref.incident_number or "-"
    state = ref.state or "-"
    category = ref.category or "-"
    short = ref.short_description.replace("\n", " ").strip() or "-"
    base = f"{idx:6d}  {incident:12s}  {state:10s}  {category:16.16s}  {short}"
    if len(base) <= width:
        return base
    return base[: max(0, width - 1)] + "…"


def _load_record(ref: RecordRef) -> dict[str, Any] | None:
    if ref.offset is None:
        # gzip: re-scan to the recorded line number.
        for _, line_no, rec in _iter_records_from_file(ref.path):
            if line_no == ref.line_no and isinstance(rec, dict):
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
    incident = _as_dict(rec.get("incident_fields"))
    discussions = _as_dict(rec.get("discussions"))

    def add_kv(key: str) -> None:
        v = rec.get(key)
        if v is None or v == "":
            v = incident.get(key)
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
        "category",
        "subcategory",
        "u_resource",
        "close_code",
        "source_path",
    ):
        add_kv(k)

    lines.append("")

    def add_block(label: str, value: Any) -> None:
        s = _norm(value)
        if not s:
            return
        lines.append(f"{label}:")
        for w in _wrap_lines(s, width=max(10, width - 2)):
            lines.append("  " + w)
        lines.append("")

    def join_discussions(key: str) -> str:
        entries = discussions.get(key)
        if not isinstance(entries, list):
            return ""
        out: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            t = _norm(entry.get("text"))
            if t:
                out.append(t)
        return "\n\n".join(out)

    add_block("short_description", incident.get("short_description") or rec.get("short_description"))
    add_block("description", incident.get("description") or rec.get("description"))
    add_block("close_notes", incident.get("close_notes") or rec.get("close_notes"))
    add_block("customer_facing_comments", join_discussions("customer_facing_comments"))
    add_block("internal_work_notes", join_discussions("internal_work_notes"))

    if lines and lines[-1] == "":
        lines.pop()
    return lines


def run_tui(args: argparse.Namespace) -> int:
    # Import curses lazily so non-interactive uses don't require it.
    try:
        import curses
    except Exception as e:
        raise SystemExit(f"curses is required for tui mode: {e}")

    refs = _index_tickets(
        args.input,
        recursive=not bool(args.no_recursive),
        limit=args.limit_index,
    )

    if not refs:
        print("No records found.")
        return 2

    query = args.query or ""
    visible: list[int] = list(range(len(refs)))
    selected_pos = 0
    top = 0

    def apply_filter() -> None:
        nonlocal visible, selected_pos, top
        q = query.strip()
        q_l = q.lower()
        out: list[int] = []

        # Support simple "field:value" filters mixed with free-text tokens.
        # Examples:
        # - state:closed vpn
        # - category:network subcategory:wifi
        filters: list[tuple[str, str]] = []
        terms: list[str] = []
        for tok in (t for t in q.split() if t.strip()):
            if ":" in tok and not tok.startswith(":") and not tok.endswith(":"):
                k, v = tok.split(":", 1)
                k = k.strip().lower()
                v = v.strip().lower()
                if k and v:
                    filters.append((k, v))
                    continue
            terms.append(tok.lower())

        for idx, r in enumerate(refs):
            if not q:
                out.append(idx)
                continue

            ok = True
            for k, v in filters:
                if k in ("incident", "incident_number", "number"):
                    ok = v in (r.incident_number or "").lower()
                elif k in ("sys_id", "sysid"):
                    ok = v in (r.sys_id or "").lower()
                elif k == "state":
                    ok = v in (r.state or "").lower()
                elif k == "category":
                    ok = v in (r.category or "").lower()
                elif k == "subcategory":
                    ok = v in (r.subcategory or "").lower()
                elif k in ("resource", "u_resource"):
                    ok = v in (r.u_resource or "").lower()
                elif k in ("close_code", "closecode"):
                    ok = v in (r.close_code or "").lower()
                elif k in ("source", "source_path", "path"):
                    ok = v in (r.source_path or "").lower()
                else:
                    ok = v in r.search_blob
                if not ok:
                    break
            if not ok:
                continue
            if all(t in r.search_blob for t in terms):
                out.append(idx)
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
        max_w = min(max(20, w - 1), int(args.width))

        title = f"Tickets: {len(refs)}  Visible: {len(visible)}"
        if query:
            title += f"  Search: {query!r}"
        stdscr.addstr(0, 0, title[: max(0, max_w)])

        help_line = "↑/↓ move  Home/End start/end  PgUp/PgDn page  Enter details  / search  q quit"
        stdscr.addstr(h - 1, 0, help_line[: max(0, max_w)])

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
            line = _format_row(refs[idx], idx, width=max_w)
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
            max_w = min(max(20, w - 1), int(args.width))
            header = _format_row(ref, -1, width=max_w).replace("-1", "sel")
            stdscr.addstr(0, 0, header[: max(0, max_w)])
            help_line = "Esc back  ↑/↓ scroll  g/G start/end  PgUp/PgDn page  j toggle json  / search  q quit"
            stdscr.addstr(h - 1, 0, help_line[: max(0, max_w)])

            lines = _record_to_detail_lines(rec, width=max_w, raw_json=raw_json)
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
        nonlocal selected_pos, top, query
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
        description="Arrow-key TUI for browsing ticket JSONL shards.",
    )
    parser.add_argument(
        "--input",
        default="/mscratch/sd/a/asnaylor/servicenow_incidents_jsonl_filtered/",
        help="Ticket JSONL directory/file to browse (default: /mscratch/.../servicenow_incidents_jsonl_filtered/).",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive file discovery.")
    parser.add_argument("--query", default=None, help="Initial search query (supports field:value tokens).")
    parser.add_argument("--width", type=int, default=160, help="Max width for rendered rows/details.")
    parser.add_argument("--limit-index", type=int, default=None, help="Limit indexed records (debug).")
    args = parser.parse_args(argv)

    return int(run_tui(args))


if __name__ == "__main__":
    raise SystemExit(main())

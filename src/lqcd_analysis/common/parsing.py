from __future__ import annotations

from pathlib import Path


def parse_optional_int(value: str) -> int | None:
    """Parse an integer value, allowing "auto" to become None.

    Args:
        value: String to parse, can be "auto" or an integer string.

    Returns:
        Integer if value is a valid integer, None if value is "auto".

    Raises:
        ValueError: If value is not "auto" and not a valid integer.
    """
    if value.lower() == "auto":
        return None
    return int(value)


def parse_tsrange(entries: dict[str, list[str]], nt: int) -> tuple[int, int]:
    """Parse the tsrange (time-slice range) from configuration entries.

    Args:
        entries: Dictionary of configuration key-value pairs.
        nt: Total number of time slices.

    Returns:
        Tuple of (start_t, end_t). If "tsrange" is not in entries,
        returns (0, max(0, nt // 2 - 1)).

    Raises:
        ValueError: If "tsrange" values cannot be parsed as integers.
    """
    if "tsrange" in entries:
        return int(entries["tsrange"][0]), int(entries["tsrange"][1])
    return 0, max(0, nt // 2 - 1)


def parse_int_list_or_range(
    entries: dict[str, list[str]],
    list_key: str,
    range_key: str
) -> tuple[int, ...]:
    """Parse either a list of integers or a range specification.

    Args:
        entries: Dictionary of configuration key-value pairs.
        list_key: Key for a space-separated list of integers.
        range_key: Key for a range specification "start stop".

    Returns:
        Tuple of integers, either from the list or generated from range.

    Raises:
        ValueError: If neither key is present, or values cannot be parsed.
    """
    if list_key in entries:
        return tuple(int(item) for item in entries[list_key])
    if range_key in entries:
        start, stop = (int(item) for item in entries[range_key][:2])
        step = 1 if stop >= start else -1
        return tuple(range(start, stop + step, step))
    raise ValueError(f"missing required key: {list_key} or {range_key}")


def parse_bool(value: str) -> bool:
    """Parse a boolean value from string.

    Args:
        value: String to parse as boolean.

    Returns:
        True for "true", "1", "yes", "y"; False for "false", "0", "no", "n".

    Raises:
        ValueError: If value cannot be parsed as boolean.
    """
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")


def parse_fold_t(entries: dict[str, list[str]]) -> str:
    """Parse the fold_t option for temporal boundary conditions.

    Args:
        entries: Dictionary of configuration key-value pairs.

    Returns:
        One of: "none", "periodic", "antiperiodic".

    Raises:
        ValueError: If fold_t value is invalid.
    """
    if "fold_t" not in entries:
        return "none"

    value = entries["fold_t"][0].strip().lower()
    if value in {"true", "periodic"}:
        return "periodic"
    if value in {"false", "none"}:
        return "none"
    if value == "antiperiodic":
        return "antiperiodic"
    raise ValueError("fold_t must be one of: false, none, true, periodic, antiperiodic")


def load_fit_window_table(
    path: str | Path,
) -> dict[tuple[str | None, int], tuple[int, int]]:
    """Load a fit window table with rows of the form `pz tmin tmax` or `gm pz tmin tmax`."""
    file_path = Path(path)
    fit_windows: dict[tuple[str | None, int], tuple[int, int]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0].lower() in {"pz", "gm"}:
                continue

            if len(tokens) == 3:
                gm = None
                pz_text, tmin_text, tmax_text = tokens
            elif len(tokens) >= 4:
                gm = tokens[0]
                pz_text, tmin_text, tmax_text = tokens[1:4]
            else:
                raise ValueError(
                    f"invalid fit_window row at {file_path}:{line_number}; "
                    "expected: pz tmin tmax or gm pz tmin tmax"
                )

            pz = int(pz_text)
            tmin = int(tmin_text)
            tmax = int(tmax_text)
            if tmax < tmin:
                raise ValueError(
                    f"invalid fit window at {file_path}:{line_number}; tmax must be >= tmin"
                )
            fit_windows[(gm, pz)] = (tmin, tmax)
    return fit_windows

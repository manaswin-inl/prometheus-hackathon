"""
plotly_tool.py - a declarative, LLM-friendly plotting tool built on Plotly.

The LLM never writes Plotly code. It fills in a JSON "spec" (see TOOL_SCHEMA),
and `plot_from_spec(df, spec)` turns that spec into a figure. Errors come back
as plain-English messages (with column-name suggestions) so the LLM can fix
its spec and retry.

Typical loop:
    1. Send the LLM `describe_dataframe(df)` so it knows the columns.
    2. Give it TOOL_SCHEMA as a tool definition.
    3. When it calls the tool, run `run_plot_tool(df, tool_input)` and send the
       returned dict back as the tool result.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
XY_TYPES = {
    "scatter",
    "line",
    "area",
    "bar",
    "bubble",
    "histogram",
    "box",
    "violin",
    "heatmap",
}
DOMAIN_TYPES = {"pie", "donut"}  # these need a "domain" subplot cell, not x/y axes
SUPPORTED_TYPES = XY_TYPES | DOMAIN_TYPES
AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "count",
    "min",
    "max",
    "nunique",
    "first",
    "last",
    "std",
}
FILTER_OPS = {
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not_in",
    "contains",
    "between",
    "notnull",
}
WEBGL_THRESHOLD = 10_000  # switch scatter/line to WebGL above this many points
PALETTE = qualitative.Plotly + qualitative.D3
SAVE_FORMATS = ("html", "png", "svg", "pdf")  # png/svg/pdf need kaleido

# Styling vocabularies
MARKER_SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "octagon",
    "star",
    "hourglass",
    "bowtie",
)  # any of these + "-open" for hollow markers
SYMBOL_CYCLE = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "x",
    "cross",
    "star",
    "triangle-down",
    "pentagon",
    "hexagon",
)
LINE_DASHES = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")
BAR_PATTERNS = ("/", "\\", "x", "-", "|", "+", ".")
SYMBOL_TYPES = {"scatter", "line", "bubble"}  # marker_symbol, symbol_by, marker_size
LINE_TYPES = {"line", "area"}  # line_dash, line_width
PATTERN_TYPES = {"bar", "histogram"}  # pattern, pattern_by
COLOR_BY_TYPES = {"scatter", "line", "bubble", "bar"}  # continuous color scale


class SpecError(Exception):
    """Raised when the spec cannot be turned into a figure."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def describe_dataframe(df: pd.DataFrame, max_unique: int = 10) -> str:
    """Compact schema summary to put in the LLM's context before it plots."""
    lines = [f"DataFrame with {len(df)} rows and {len(df.columns)} columns:"]
    for col in df.columns:
        s = df[col]
        info = f"- {col!r} ({s.dtype}, {s.isna().sum()} nulls"
        if pd.api.types.is_numeric_dtype(s):
            info += f", min={s.min()}, max={s.max()})"
        else:
            uniq = s.dropna().unique()
            sample = ", ".join(map(str, uniq[:max_unique]))
            more = (
                f" ... +{len(uniq) - max_unique} more" if len(uniq) > max_unique else ""
            )
            info += f", {len(uniq)} unique: {sample}{more})"
        lines.append(info)
    return "\n".join(lines)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _resolve_col(df: pd.DataFrame, name: str, where: str) -> str:
    """Return the real column name, tolerating case/whitespace slips."""
    if name in df.columns:
        return name
    lookup = {str(c).strip().lower(): c for c in df.columns}
    key = str(name).strip().lower()
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(
        str(name), [str(c) for c in df.columns], n=3, cutoff=0.5
    )
    hint = (
        f" Did you mean: {', '.join(close)}?"
        if close
        else f" Available columns: {list(df.columns)}"
    )
    raise SpecError(f"{where}: column '{name}' not found.{hint}")


def _apply_filters(df: pd.DataFrame, filters: list[dict], where: str) -> pd.DataFrame:
    """Structured filters (no eval/query strings, so nothing executable from the LLM)."""
    for i, f in enumerate(filters or []):
        col = _resolve_col(df, f.get("column"), f"{where}.filters[{i}]")
        op, val = f.get("op", "=="), f.get("value")
        s = df[col]
        if op not in FILTER_OPS:
            raise SpecError(
                f"{where}.filters[{i}]: unknown op '{op}'. Use one of {sorted(FILTER_OPS)}."
            )
        if op == "==":
            mask = s == val
        elif op == "!=":
            mask = s != val
        elif op == ">":
            mask = s > val
        elif op == ">=":
            mask = s >= val
        elif op == "<":
            mask = s < val
        elif op == "<=":
            mask = s <= val
        elif op == "in":
            mask = s.isin(_as_list(val))
        elif op == "not_in":
            mask = ~s.isin(_as_list(val))
        elif op == "contains":
            mask = s.astype(str).str.contains(str(val), case=False, na=False)
        elif op == "between":
            if not isinstance(val, list) or len(val) != 2:
                raise SpecError(
                    f"{where}.filters[{i}]: 'between' needs value=[low, high]."
                )
            mask = s.between(val[0], val[1])
        else:  # notnull
            mask = s.notna()
        df = df[mask]
    return df


# --------------------------------------------------------------------------- #
# Styling (colors, marker shapes, line styles, bar patterns)
# --------------------------------------------------------------------------- #
def _check_symbol(sym: str, where: str) -> None:
    base = re.sub(r"(-open)?(-dot)?$", "", str(sym))
    if base not in MARKER_SYMBOLS:
        raise SpecError(
            f"{where}: unknown marker symbol '{sym}'. Use one of {list(MARKER_SYMBOLS)} "
            f"(add '-open' for a hollow marker, e.g. 'circle-open')."
        )


def _check_choice(value: str, allowed: tuple, where: str, what: str) -> None:
    if value not in allowed:
        raise SpecError(
            f"{where}: unknown {what} '{value}'. Use one of {list(allowed)}."
        )


def _str_keys(mapping: dict | None) -> dict:
    return {str(k): v for k, v in (mapping or {}).items()}


class _Styles:
    """
    Keeps colors, marker symbols, bar patterns and legend entries consistent across
    the whole figure, so "North" gets the same color/shape in every subplot.
    Lookup order: trace-level map -> figure-level map -> next unused value in the cycle.
    """

    def __init__(self, color_map=None, symbol_map=None, pattern_map=None):
        self.maps = {
            "color": _str_keys(color_map),
            "symbol": _str_keys(symbol_map),
            "pattern": _str_keys(pattern_map),
        }
        for k, v in self.maps["symbol"].items():
            _check_symbol(v, f"spec.symbol_map['{k}']")
        for k, v in self.maps["pattern"].items():
            _check_choice(v, BAR_PATTERNS, f"spec.pattern_map['{k}']", "pattern")
        self.cycles = {
            "color": PALETTE,
            "symbol": SYMBOL_CYCLE,
            "pattern": BAR_PATTERNS,
        }
        self.auto: dict[str, dict[str, str]] = {
            "color": {},
            "symbol": {},
            "pattern": {},
        }
        self.legend_shown: set[str] = set()
        self.colorbar_shown: set[str] = set()

    def pick(self, kind: str, keys: list, trace_map: dict | None = None) -> str:
        keys = [str(k) for k in keys if k is not None]
        for mapping in (trace_map or {}, self.maps[kind]):
            for k in keys:
                if k in mapping:
                    return mapping[k]
        auto = self.auto[kind]
        if keys[0] not in auto:
            cycle = self.cycles[kind]
            taken = set(self.maps[kind].values()) | set(auto.values())
            free = [c for c in cycle if c not in taken]
            auto[keys[0]] = free[0] if free else cycle[len(auto) % len(cycle)]
        return auto[keys[0]]

    def show_legend(self, name: str) -> bool:
        if name in self.legend_shown:
            return False
        self.legend_shown.add(name)
        return True

    def show_colorbar(self, col: str) -> bool:
        if col in self.colorbar_shown:
            return False
        self.colorbar_shown.add(col)
        return True


def _validate_trace(t: dict, where: str) -> None:
    ttype = t.get("type")
    if ttype not in SUPPORTED_TYPES:
        raise SpecError(
            f"{where}: unsupported type '{ttype}'. Use one of {sorted(SUPPORTED_TYPES)}."
        )
    if t.get("agg") and t["agg"] not in AGGREGATIONS:
        raise SpecError(
            f"{where}: unknown agg '{t['agg']}'. Use one of {sorted(AGGREGATIONS)}."
        )
    has_y = bool(_as_list(t.get("y")))
    if ttype in {"scatter", "line", "area", "bubble"} and not (t.get("x") and has_y):
        raise SpecError(f"{where}: '{ttype}' needs both 'x' and 'y'.")
    if ttype == "bar" and not t.get("x"):
        raise SpecError(
            f"{where}: 'bar' needs 'x' (and 'y', or agg='count' to count rows)."
        )
    if ttype == "bar" and not has_y and t.get("agg") != "count":
        raise SpecError(f"{where}: 'bar' without 'y' needs agg='count'.")
    if ttype == "bubble" and not t.get("size"):
        raise SpecError(f"{where}: 'bubble' needs a 'size' column.")
    if ttype == "histogram" and not (t.get("x") or has_y):
        raise SpecError(f"{where}: 'histogram' needs 'x' (or 'y' for horizontal).")
    if ttype in {"box", "violin"} and not has_y:
        raise SpecError(
            f"{where}: '{ttype}' needs 'y' (the numeric column); 'x' is optional categories."
        )
    if ttype in DOMAIN_TYPES and not t.get("labels"):
        raise SpecError(f"{where}: '{ttype}' needs 'labels' (and optionally 'values').")


def _validate_style(df: pd.DataFrame, t: dict, where: str, warnings: list[str]) -> None:
    """Check style values and drop options that don't apply to this chart type (with a warning)."""
    ttype = t["type"]

    def drop(keys: tuple, applies_to: set) -> None:
        if ttype in applies_to:
            return
        used = [k for k in keys if t.get(k) not in (None, [], {})]
        for k in used:
            t.pop(k)
        if used:
            warnings.append(
                f"{where}: {', '.join(used)} ignored - only used by {sorted(applies_to)} charts."
            )

    drop(("marker_symbol", "symbol_by", "symbol_map", "marker_size"), SYMBOL_TYPES)
    drop(("line_dash", "line_width"), LINE_TYPES)
    drop(("pattern", "pattern_by", "pattern_map"), PATTERN_TYPES)
    drop(("color_by",), COLOR_BY_TYPES)

    if t.get("marker_symbol"):
        _check_symbol(t["marker_symbol"], f"{where}.marker_symbol")
    for k, v in _str_keys(t.get("symbol_map")).items():
        _check_symbol(v, f"{where}.symbol_map['{k}']")
    if t.get("line_dash"):
        _check_choice(t["line_dash"], LINE_DASHES, f"{where}.line_dash", "line_dash")
    if t.get("pattern"):
        _check_choice(t["pattern"], BAR_PATTERNS, f"{where}.pattern", "pattern")
    for k, v in _str_keys(t.get("pattern_map")).items():
        _check_choice(v, BAR_PATTERNS, f"{where}.pattern_map['{k}']", "pattern")

    col = t.get("color_by")
    if col:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if t.get("group_by"):
                raise SpecError(
                    f"{where}: color_by '{col}' is not numeric. Use group_by for categories "
                    f"(this trace already has group_by)."
                )
            t["group_by"] = [col]
            t.pop("color_by")
            warnings.append(
                f"{where}: color_by '{col}' is categorical, so it was used as group_by "
                f"(one color per category with a legend)."
            )
        elif t.get("agg") and col not in _as_list(t.get("y")):
            raise SpecError(
                f"{where}: with agg, color_by must be one of the y columns "
                f"(the aggregated values), not '{col}'."
            )


def _aggregate(data: pd.DataFrame, x: str, ycols: list[str], agg: str) -> pd.DataFrame:
    if agg == "count" and not ycols:
        return (
            data.groupby(x, as_index=False, dropna=False)
            .size()
            .rename(columns={"size": "count"})
        )
    return data.groupby(x, as_index=False, dropna=False)[ycols].agg(agg)


def _sort(
    data: pd.DataFrame, t: dict, x: str | None, default_sort_x: bool
) -> pd.DataFrame:
    sort_by = t.get("sort_by")
    ascending = t.get("ascending", True)
    if sort_by:
        col = x if sort_by == "x" else sort_by
        if sort_by == "y":
            col = _as_list(t.get("y"))[0] if t.get("y") else "count"
        if col in data.columns:
            return data.sort_values(col, ascending=ascending)
    if default_sort_x and x and x in data.columns:
        return data.sort_values(x)
    return data


def _series(t: dict, split: dict, y: str, multi_y: bool, styles: _Styles) -> dict:
    """Legend name, color, symbol and pattern for one series."""
    label = split.get("label")
    if label:
        name = f"{label} - {y}" if multi_y else label
    else:
        name = (
            f"{t['name']} - {y}" if t.get("name") and multi_y else (t.get("name") or y)
        )

    base = split.get("color_label") or t.get("name")
    key = f"{base} - {y}" if base and multi_y else (base or y)
    color = t.get("color") or styles.pick(
        "color", [key, split.get("color_label"), y], t.get("color_map")
    )
    return {
        "name": name,
        "color": color,
        "symbol": split.get("symbol") or t.get("marker_symbol"),
        "pattern": split.get("pattern") or t.get("pattern"),
        "legend": dict(
            name=name, legendgroup=name, showlegend=styles.show_legend(name)
        ),
    }


def _marker(
    t: dict, data: pd.DataFrame, s: dict, styles: _Styles, with_size: bool = True
) -> dict:
    m: dict[str, Any] = {"color": s["color"]}
    if s["symbol"]:
        m["symbol"] = s["symbol"]
    if with_size and t.get("marker_size"):
        m["size"] = t["marker_size"]
    if s["pattern"]:
        m["pattern"] = {"shape": s["pattern"]}
    col = t.get("color_by")
    if col and col in data.columns:
        lo, hi = t["_color_range"]
        m.update(
            color=data[col],
            colorscale=t.get("colorscale", "Viridis"),
            cmin=lo,
            cmax=hi,
            showscale=styles.show_colorbar(col),
            colorbar=dict(title=dict(text=col)),
        )
    return m


def _xy_traces(data: pd.DataFrame, t: dict, split: dict, styles: _Styles) -> list:
    """scatter / line / area / bar / bubble - one trace per y column."""
    ttype = t["type"]
    x = t["x"]
    ycols = _as_list(t.get("y"))
    agg = t.get("agg")
    if agg:
        if t.get("size") or t.get("text"):
            raise SpecError(
                f"trace '{t.get('name') or ycols}': 'size'/'text' can't be combined with 'agg'."
            )
        data = _aggregate(data, x, ycols, agg)
        if not ycols:
            ycols = ["count"]
    data = _sort(data, t, x, default_sort_x=ttype in {"line", "area"})
    big = len(data) > WEBGL_THRESHOLD

    traces = []
    for y in ycols:
        s = _series(t, split, y, len(ycols) > 1, styles)
        hover_text = data[t["text"]] if t.get("text") else None
        common = dict(opacity=t.get("opacity"), text=hover_text, **s["legend"])

        if ttype == "bar":
            horiz = t.get("orientation") == "h"
            traces.append(
                go.Bar(
                    x=data[y] if horiz else data[x],
                    y=data[x] if horiz else data[y],
                    orientation="h" if horiz else "v",
                    marker=_marker(t, data, s, styles),
                    **common,
                )
            )
            continue

        line = {
            k: v
            for k, v in {
                "color": s["color"],
                "dash": t.get("line_dash"),
                "width": t.get("line_width"),
            }.items()
            if v is not None
        }
        Scatter = go.Scattergl if big and ttype != "area" else go.Scatter
        kw: dict[str, Any] = dict(x=data[x], y=data[y], **common)
        if ttype == "scatter":
            kw.update(mode="markers", marker=_marker(t, data, s, styles))
        elif ttype == "line":
            show_markers = any(
                t.get(k)
                for k in ("markers", "marker_symbol", "marker_size", "color_by")
            ) or bool(split.get("symbol"))
            kw.update(
                mode="lines+markers" if show_markers else "lines",
                line=line,
                marker=_marker(t, data, s, styles),
            )
        elif ttype == "area":
            kw.update(mode="lines", line=line)
            if t.get("stacked"):
                kw["stackgroup"] = "stack"
            else:
                kw["fill"] = "tozeroy"
        elif ttype == "bubble":
            sizes = data[t["size"]]
            max_px = t.get("marker_size") or 40  # marker_size = largest bubble diameter
            sizeref = 2.0 * t.get("_size_max", sizes.max() or 1) / (max_px**2)
            m = _marker(t, data, s, styles, with_size=False)
            m.update(size=sizes, sizemode="area", sizeref=sizeref)
            kw.update(mode="markers", marker=m)
        traces.append(Scatter(**kw))
    return traces


def _dist_traces(data: pd.DataFrame, t: dict, split: dict, styles: _Styles) -> list:
    """histogram / box / violin."""
    ttype = t["type"]
    if ttype == "histogram":
        col = t.get("x") or _as_list(t.get("y"))[0]
        s = _series(t, split, col, False, styles)
        horiz = not t.get("x")
        return [
            go.Histogram(
                **({"y": data[col]} if horiz else {"x": data[col]}),
                nbinsx=None if horiz else t.get("nbins"),
                nbinsy=t.get("nbins") if horiz else None,
                histnorm=t.get("histnorm"),
                marker=_marker(t, data, s, styles, with_size=False),
                opacity=t.get("opacity"),
                **s["legend"],
            )
        ]

    traces = []
    ycols = _as_list(t.get("y"))
    for y in ycols:
        s = _series(t, split, y, len(ycols) > 1, styles)
        common = dict(
            y=data[y],
            x=data[t["x"]] if t.get("x") else None,
            marker_color=s["color"],
            opacity=t.get("opacity"),
            **s["legend"],
        )
        if ttype == "box":
            traces.append(go.Box(boxpoints=t.get("points", "outliers"), **common))
        else:
            traces.append(
                go.Violin(
                    box_visible=True,
                    meanline_visible=True,
                    points=t.get("points", False),
                    **common,
                )
            )
    return traces


def _heatmap_trace(df: pd.DataFrame, t: dict, where: str):
    """Pivot heatmap (x, y, z) or a correlation matrix when z is omitted."""
    if t.get("x") and t.get("y") and t.get("z"):
        x, y = _resolve_col(df, t["x"], where), _resolve_col(
            df, _as_list(t["y"])[0], where
        )
        z = _resolve_col(df, t["z"], where)
        pivot = df.pivot_table(
            index=y, columns=x, values=z, aggfunc=t.get("agg") or "mean"
        )
        return go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale=t.get("colorscale", "Viridis"),
            colorbar=dict(title=dict(text=z)),
            name=z,
        )
    cols = [
        _resolve_col(df, c, where) for c in _as_list(t.get("columns"))
    ] or df.select_dtypes("number").columns.tolist()
    if len(cols) < 2:
        raise SpecError(
            f"{where}: correlation heatmap needs at least 2 numeric columns."
        )
    corr = df[cols].corr(method=t.get("method", "pearson"))
    return go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        zmin=-1,
        zmax=1,
        colorscale=t.get("colorscale", "RdBu"),
        texttemplate="%{z:.2f}",
        name="correlation",
    )


def _pie_trace(data: pd.DataFrame, t: dict, styles: _Styles):
    labels = t["labels"]
    if t.get("values"):
        grouped = data.groupby(labels, as_index=False)[t["values"]].agg(
            t.get("agg") or "sum"
        )
        vals = grouped[t["values"]]
    else:
        grouped = data[labels].value_counts().reset_index()
        grouped.columns = [labels, "count"]
        vals = grouped["count"]
    # Slice colors come from the shared styles, so "North" matches its line in other subplots.
    colors = [
        styles.pick("color", [lab], t.get("color_map")) for lab in grouped[labels]
    ]
    return go.Pie(
        labels=grouped[labels],
        values=vals,
        name=t.get("name") or labels,
        marker=dict(colors=colors),
        hole=t.get("hole", 0.4 if t["type"] == "donut" else 0),
        textinfo=t.get("textinfo", "percent+label"),
        sort=True,
    )


def _build_traces(
    df: pd.DataFrame, t: dict, where: str, styles: _Styles, warnings: list[str]
) -> list:
    _validate_trace(t, where)
    # Resolve every column reference up front so errors are clear.
    if t["type"] != "heatmap":
        for key in ("x", "size", "text", "labels", "values", "color_by"):
            if t.get(key):
                t[key] = _resolve_col(df, t[key], f"{where}.{key}")
        if t.get("y"):
            t["y"] = [_resolve_col(df, c, f"{where}.y") for c in _as_list(t["y"])]
    for key in ("group_by", "symbol_by", "pattern_by"):
        t[key] = [_resolve_col(df, c, f"{where}.{key}") for c in _as_list(t.get(key))]
    for key in ("color_map", "symbol_map", "pattern_map"):
        t[key] = _str_keys(t.get(key))
    _validate_style(df, t, where, warnings)

    data = _apply_filters(df, t.get("filters"), where)
    if data.empty:
        raise SpecError(f"{where}: filters removed every row.")
    if t.get("color_by"):  # one shared color scale for every split of this trace
        vals = data[t["color_by"]]
        if t.get("agg"):
            vals = _aggregate(data, t["x"], _as_list(t["y"]), t["agg"])[t["color_by"]]
        t["_color_range"] = (float(vals.min()), float(vals.max()))
    if (
        t["type"] == "bubble"
    ):  # one bubble scale for every split, so sizes are comparable
        t["_size_max"] = float(data[t["size"]].max() or 1)

    if t["type"] == "heatmap":
        return [_heatmap_trace(data, t, where)]
    if t["type"] in DOMAIN_TYPES:
        return [_pie_trace(data, t, styles)]

    builder = (
        _dist_traces if t["type"] in {"histogram", "box", "violin"} else _xy_traces
    )
    group_by, symbol_by, pattern_by = (
        t.get("group_by", []),
        t.get("symbol_by", []),
        t.get("pattern_by", []),
    )
    split_cols = list(dict.fromkeys(group_by + symbol_by + pattern_by))
    if not split_cols:
        return builder(data, t, {}, styles)

    # One trace per combination: group_by -> color, symbol_by -> marker shape, pattern_by -> bar fill.
    traces = []
    for key, sub in data.groupby(split_cols, sort=True, dropna=False):
        values = dict(zip(split_cols, key if isinstance(key, tuple) else (key,)))

        def label_for(cols):
            return " / ".join(str(values[c]) for c in cols) if cols else None

        color_label, symbol_label, pattern_label = (
            label_for(group_by),
            label_for(symbol_by),
            label_for(pattern_by),
        )
        label = " / ".join(
            dict.fromkeys(p for p in (color_label, symbol_label, pattern_label) if p)
        )
        if t.get("name"):
            label = f"{t['name']}: {label}"
        split = {
            "label": label,
            "color_label": color_label,
            "symbol": (
                styles.pick("symbol", [symbol_label], t["symbol_map"])
                if symbol_label
                else None
            ),
            "pattern": (
                styles.pick("pattern", [pattern_label], t["pattern_map"])
                if pattern_label
                else None
            ),
        }
        traces.extend(builder(sub, t, split, styles))
    return traces


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def plot_from_spec(df: pd.DataFrame, spec: dict) -> tuple[go.Figure, list[str]]:
    """Build a Plotly figure from a declarative spec. Returns (figure, warnings)."""
    if df is None or df.empty:
        raise SpecError("The DataFrame is empty.")
    traces_spec = spec.get("traces") or []
    if not traces_spec:
        raise SpecError("spec.traces must contain at least one trace.")

    rows, cols = int(spec.get("rows", 1)), int(spec.get("cols", 1))
    warnings: list[str] = []

    # Work out what each subplot cell needs (xy vs domain, secondary y-axis).
    cell_kind: dict[tuple[int, int], str] = {}
    secondary: set[tuple[int, int]] = set()
    for i, t in enumerate(traces_spec):
        r, c = int(t.get("row", 1)), int(t.get("col", 1))
        if not (1 <= r <= rows and 1 <= c <= cols):
            raise SpecError(
                f"traces[{i}]: row={r}, col={c} is outside the {rows}x{cols} grid. "
                f"Increase spec.rows/spec.cols or fix the position."
            )
        kind = "domain" if t.get("type") in DOMAIN_TYPES else "xy"
        if cell_kind.get((r, c), kind) != kind:
            raise SpecError(
                f"traces[{i}]: can't mix pie/donut with x/y charts in subplot ({r}, {c})."
            )
        cell_kind[(r, c)] = kind
        if t.get("secondary_y"):
            secondary.add((r, c))

    specs = [
        [
            (
                {"type": "domain"}
                if cell_kind.get((r, c)) == "domain"
                else {"secondary_y": (r, c) in secondary}
            )
            for c in range(1, cols + 1)
        ]
        for r in range(1, rows + 1)
    ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=spec.get("subplot_titles"),
        shared_xaxes=spec.get("shared_xaxes", False),
        shared_yaxes=spec.get("shared_yaxes", False),
        vertical_spacing=spec.get("vertical_spacing"),
        horizontal_spacing=spec.get("horizontal_spacing"),
    )

    styles = _Styles(
        spec.get("color_map"), spec.get("symbol_map"), spec.get("pattern_map")
    )
    axis_cols: dict[tuple[int, int], dict[str, set]] = {}
    for i, t in enumerate(traces_spec):
        t = dict(t)  # don't mutate the caller's spec
        r, c = int(t.get("row", 1)), int(t.get("col", 1))
        where = f"traces[{i}]"
        built = _build_traces(df, t, where, styles, warnings)
        if not built:
            warnings.append(f"{where}: produced no traces.")
        for tr in built:
            if cell_kind[(r, c)] == "domain":
                fig.add_trace(tr, row=r, col=c)
            elif (r, c) in secondary:
                fig.add_trace(tr, row=r, col=c, secondary_y=bool(t.get("secondary_y")))
            else:
                fig.add_trace(tr, row=r, col=c)
        if cell_kind[(r, c)] == "xy" and t["type"] != "heatmap":
            ax = axis_cols.setdefault((r, c), {"x": set(), "y": set()})
            if t.get("x"):
                ax["x"].add(t["x"])
            if not t.get("secondary_y"):
                ax["y"].update(
                    _as_list(t.get("y")) or ([t["agg"]] if t.get("agg") else [])
                )

    # Default axis titles when a subplot uses a single, unambiguous column.
    for (r, c), ax in axis_cols.items():
        if len(ax["x"]) == 1:
            fig.update_xaxes(title_text=next(iter(ax["x"])), row=r, col=c)
        if len(ax["y"]) == 1:
            fig.update_yaxes(
                title_text=next(iter(ax["y"])),
                row=r,
                col=c,
                **({"secondary_y": False} if (r, c) in secondary else {}),
            )

    # Explicit per-subplot axis settings override the defaults.
    for a in spec.get("axes") or []:
        r, c = int(a.get("row", 1)), int(a.get("col", 1))
        if cell_kind.get((r, c)) != "xy":
            warnings.append(
                f"axes entry for ({r}, {c}) ignored: that cell has no x/y chart."
            )
            continue
        sec_kw = {"secondary_y": False} if (r, c) in secondary else {}
        x_kw = {
            k: v
            for k, v in {
                "title_text": a.get("x_title"),
                "range": a.get("x_range"),
                "type": "log" if a.get("log_x") else None,
                "tickformat": a.get("x_tickformat"),
            }.items()
            if v is not None
        }
        y_kw = {
            k: v
            for k, v in {
                "title_text": a.get("y_title"),
                "range": a.get("y_range"),
                "type": "log" if a.get("log_y") else None,
                "tickformat": a.get("y_tickformat"),
            }.items()
            if v is not None
        }
        if x_kw:
            fig.update_xaxes(row=r, col=c, **x_kw)
        if y_kw:
            fig.update_yaxes(row=r, col=c, **sec_kw, **y_kw)
        if a.get("y2_title") and (r, c) in secondary:
            fig.update_yaxes(title_text=a["y2_title"], row=r, col=c, secondary_y=True)

    fig.update_layout(
        title=dict(text=spec.get("title"), x=0.5) if spec.get("title") else None,
        template=spec.get("template", "plotly_white"),
        height=spec.get("height", max(450, 380 * rows)),
        width=spec.get("width"),
        barmode=spec.get("barmode", "group"),
        hovermode=spec.get("hovermode", "closest"),
        showlegend=spec.get("show_legend", True),
        legend_title_text=spec.get("legend_title"),
    )
    if styles.colorbar_shown and spec.get("show_legend", True):
        # A color_by colorbar sits on the right, so move the legend underneath to avoid overlap.
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0)
        )
    return fig, warnings


def _safe_filename(name: str | None, fallback: str = "plot") -> str:
    """Strip any extension/path and keep only safe characters."""
    stem = Path(str(name or fallback)).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return stem or fallback


def save_figure(
    fig: go.Figure,
    filename: str = "plot",
    formats: list[str] | str = "html",
    output_dir: str | Path = ".",
    scale: float = 2,
    width: int | None = None,
    height: int | None = None,
    embed_plotlyjs: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Save a figure in one or more formats: "html", "png", "svg", "pdf".

    - html is interactive; embed_plotlyjs=True makes it work offline (~4 MB bigger),
      otherwise it loads plotly.js from a CDN.
    - png/svg/pdf are static and rendered by kaleido. `scale` multiplies the pixel
      resolution of PNGs (2 = sharp on retina screens); svg/pdf are vector anyway.
    - width/height default to the figure's own layout size.

    Returns (saved, errors): {format: path} for each success, {format: message} for each failure.
    """
    formats = [f.lower().lstrip(".") for f in _as_list(formats)] or ["html"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(filename)

    width = width or fig.layout.width or 1100
    height = height or fig.layout.height or 600
    saved: dict[str, str] = {}
    errors: dict[str, str] = {}

    for fmt in dict.fromkeys(formats):  # de-duplicate, keep order
        if fmt not in SAVE_FORMATS:
            errors[fmt] = (
                f"Unsupported format '{fmt}'. Use one of {list(SAVE_FORMATS)}."
            )
            continue
        path = out / f"{stem}.{fmt}"
        if fmt == "html":
            fig.write_html(
                path,
                include_plotlyjs=True if embed_plotlyjs else "cdn",
                full_html=True,
            )
        else:
            fig.write_image(
                path,
                format=fmt,
                width=width,
                height=height,
                scale=scale if fmt == "png" else 1,
            )
        saved[fmt] = str(path)
        # except Exception as e:
        #     msg = f"{type(e).__name__}: {e}"
        #     if "kaleido" in msg.lower() or "chrome" in msg.lower():
        #         msg += (
        #             " (static export needs `pip install -U kaleido`; kaleido v1+ also needs Chrome - "
        #             "run `plotly_get_chrome` once if it isn't installed)"
        #         )
        #     errors[fmt] = msg
    return saved, errors


def run_plot_tool(
    df: pd.DataFrame,
    spec: dict,
    output_dir: str | Path = "plots",
    formats: list[str] | None = None,
    filename: str | None = None,
    embed_plotlyjs: bool = False,
) -> dict:
    """
    Tool-call handler: never raises, always returns a JSON-serialisable result for the LLM.

    Output settings come from (highest priority first): the arguments here, then
    spec["save"] chosen by the LLM, then the defaults (html, filename from the title).
    """
    try:
        fig, warnings = plot_from_spec(df, spec)
    except SpecError as e:
        return {
            "ok": False,
            "error": str(e),
            "hint": "Fix the spec and call the tool again.",
        }
    except Exception as e:  # unexpected pandas/plotly errors still go back to the LLM
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    save = spec.get("save") or {}
    saved, save_errors = save_figure(
        fig,
        filename=filename or save.get("filename") or spec.get("title") or "plot",
        formats=formats or save.get("formats") or ["html"],
        output_dir=output_dir,
        scale=save.get("scale", 2),
        width=save.get("width"),
        height=save.get("height"),
        embed_plotlyjs=embed_plotlyjs,
    )

    summary = []
    for tr in fig.data:
        arr = next(
            (
                getattr(tr, a, None)
                for a in ("x", "y", "values")
                if getattr(tr, a, None) is not None
            ),
            None,
        )
        summary.append(
            {
                "name": tr.name,
                "type": tr.type,
                "points": len(arr) if arr is not None else None,
            }
        )

    result = {
        "ok": bool(saved),
        "files": saved,
        "traces": summary,
        "warnings": warnings,
    }
    if save_errors:
        result["save_errors"] = save_errors
    if not saved:
        result["error"] = (
            "The figure was built but could not be saved in any requested format."
        )
    return result


# --------------------------------------------------------------------------- #
# Tool definition for the LLM (Anthropic format; for OpenAI wrap it as
# {"type": "function", "function": {"name", "description", "parameters": input_schema}})
# --------------------------------------------------------------------------- #
_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string"},
        "op": {"type": "string", "enum": sorted(FILTER_OPS)},
        "value": {"description": "Scalar, or a list for 'in'/'not_in'/'between'."},
    },
    "required": ["column", "op"],
}

_SYMBOL_ENUM = list(MARKER_SYMBOLS) + [f"{s}-open" for s in MARKER_SYMBOLS]

_TRACE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": sorted(SUPPORTED_TYPES)},
        "x": {
            "type": "string",
            "description": "Column for the x-axis (categories for bar/box/violin).",
        },
        "y": {
            "description": "Column name, or a list of column names to draw one trace per column.",
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
        "z": {
            "type": "string",
            "description": "heatmap only: value column to pivot on x and y. "
            "Omit x/y/z for a correlation matrix.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "heatmap only: numeric columns for the correlation matrix.",
        },
        "labels": {"type": "string", "description": "pie/donut: category column."},
        "values": {
            "type": "string",
            "description": "pie/donut: value column (omit to count rows).",
        },
        "size": {
            "type": "string",
            "description": "bubble: column controlling marker size.",
        },
        "text": {"type": "string", "description": "Column shown on hover."},
        "group_by": {
            "description": "Split into one trace per unique value (like a color legend).",
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
        "agg": {
            "type": "string",
            "enum": sorted(AGGREGATIONS),
            "description": "Aggregate y by x (per group). Use 'count' with no y to count rows.",
        },
        "filters": {"type": "array", "items": _FILTER_SCHEMA},
        "sort_by": {"type": "string", "description": "'x', 'y', or a column name."},
        "ascending": {"type": "boolean"},
        "row": {"type": "integer", "minimum": 1, "default": 1},
        "col": {"type": "integer", "minimum": 1, "default": 1},
        "secondary_y": {
            "type": "boolean",
            "description": "Plot on a right-hand y-axis.",
        },
        "name": {"type": "string", "description": "Legend name."},
        "color": {
            "type": "string",
            "description": "One fixed color for the whole trace, e.g. 'red' or "
            "'#1f77b4'. Don't combine with group_by (use color_map).",
        },
        "color_map": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Category/series -> color, e.g. {'North': 'blue', 'South': '#ff7f0e'}. "
            "Keys are group_by values, y column names, or pie labels.",
        },
        "color_by": {
            "type": "string",
            "description": "Numeric column that colors each point/bar on a gradient with a colorbar "
            "(scatter, line, bubble, bar). Set the gradient with colorscale. "
            "For categories use group_by instead.",
        },
        "marker_symbol": {
            "type": "string",
            "enum": _SYMBOL_ENUM,
            "description": "Marker shape for scatter/line/bubble.",
        },
        "symbol_by": {
            "description": "Column(s) whose categories each get a different marker shape "
            "(scatter/line/bubble). Can be combined with group_by.",
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
        "symbol_map": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": _SYMBOL_ENUM},
            "description": "symbol_by category -> marker shape, e.g. {'Actual': 'circle'}.",
        },
        "marker_size": {
            "type": "number",
            "minimum": 1,
            "description": "Marker size in px (scatter/line). For bubble: the largest bubble's size.",
        },
        "line_dash": {
            "type": "string",
            "enum": list(LINE_DASHES),
            "description": "line/area only.",
        },
        "line_width": {
            "type": "number",
            "minimum": 0,
            "description": "line/area only, in px.",
        },
        "pattern": {
            "type": "string",
            "enum": list(BAR_PATTERNS),
            "description": "Fill pattern for bar/histogram: / \\ diagonal hatch, x crosshatch, "
            "- horizontal, | vertical, + grid, . dots.",
        },
        "pattern_by": {
            "description": "Column(s) whose categories each get a different bar pattern.",
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
        "pattern_map": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": list(BAR_PATTERNS)},
            "description": "pattern_by category -> pattern.",
        },
        "orientation": {
            "type": "string",
            "enum": ["v", "h"],
            "description": "bar only.",
        },
        "stacked": {"type": "boolean", "description": "area only: stack areas."},
        "markers": {"type": "boolean", "description": "line only: show points."},
        "nbins": {"type": "integer", "description": "histogram only."},
        "histnorm": {
            "type": "string",
            "enum": ["percent", "probability", "density", "probability density"],
        },
        "points": {"description": "box/violin: 'all', 'outliers', or false."},
        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
        "colorscale": {
            "type": "string",
            "description": "Gradient for heatmap or color_by, e.g. 'Viridis', "
            "'RdBu', 'Blues', 'Plasma', 'Turbo'.",
        },
    },
    "required": ["type"],
}

TOOL_SCHEMA = {
    "name": "create_plot",
    "description": (
        "Create an interactive Plotly chart from the loaded DataFrame. Supports scatter, line, area, bar, "
        "bubble, histogram, box, violin, heatmap, pie and donut charts; multiple traces on one chart; "
        "grids of subplots (set rows/cols and give each trace a row/col); secondary y-axes; grouping "
        "(group_by), aggregation (agg) and filtering; and styling: colors per category (color_map), "
        "gradient coloring by a numeric column (color_by), marker shapes (marker_symbol, symbol_by), "
        "line styles (line_dash, line_width), marker sizes and bar fill patterns (pattern, pattern_by). Use exact column names from the DataFrame schema. "
        "pie/donut charts must be in their own subplot cell. If the result has ok=false, read the error, "
        "fix the spec, and call again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rows": {"type": "integer", "minimum": 1, "default": 1},
            "cols": {"type": "integer", "minimum": 1, "default": 1},
            "subplot_titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One per cell, left-to-right then top-to-bottom.",
            },
            "shared_xaxes": {"type": "boolean"},
            "shared_yaxes": {"type": "boolean"},
            "traces": {"type": "array", "items": _TRACE_SCHEMA, "minItems": 1},
            "axes": {
                "type": "array",
                "description": "Per-subplot axis settings.",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer"},
                        "col": {"type": "integer"},
                        "x_title": {"type": "string"},
                        "y_title": {"type": "string"},
                        "y2_title": {
                            "type": "string",
                            "description": "Secondary y-axis title.",
                        },
                        "log_x": {"type": "boolean"},
                        "log_y": {"type": "boolean"},
                        "x_range": {"type": "array", "items": {}},
                        "y_range": {"type": "array", "items": {}},
                        "x_tickformat": {"type": "string"},
                        "y_tickformat": {
                            "type": "string",
                            "description": "e.g. '$,.0f' or '.0%'",
                        },
                    },
                },
            },
            "barmode": {
                "type": "string",
                "enum": ["group", "stack", "overlay", "relative"],
            },
            "template": {
                "type": "string",
                "enum": [
                    "plotly_white",
                    "plotly",
                    "plotly_dark",
                    "ggplot2",
                    "seaborn",
                    "simple_white",
                ],
            },
            "height": {"type": "integer"},
            "width": {"type": "integer"},
            "hovermode": {
                "type": "string",
                "enum": ["closest", "x", "x unified", "y unified"],
            },
            "legend_title": {"type": "string"},
            "show_legend": {"type": "boolean"},
            "color_map": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Figure-wide category -> color, applied in every subplot "
                "(a trace's own color_map wins).",
            },
            "symbol_map": {
                "type": "object",
                "additionalProperties": {"type": "string", "enum": _SYMBOL_ENUM},
                "description": "Figure-wide category -> marker shape.",
            },
            "pattern_map": {
                "type": "object",
                "additionalProperties": {"type": "string", "enum": list(BAR_PATTERNS)},
                "description": "Figure-wide category -> bar pattern.",
            },
            "save": {
                "type": "object",
                "description": "How to save the chart. Default: interactive html only. Add png/svg/pdf "
                "when the user asks for an image, a file to put in a document/slide, or print.",
                "properties": {
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SAVE_FORMATS)},
                        "description": "One or more of html (interactive), png (image), "
                        "svg (vector image), pdf.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Descriptive snake_case name without extension, "
                        "e.g. 'monthly_revenue_by_region'.",
                    },
                    "scale": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 5,
                        "description": "PNG resolution multiplier (default 2).",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in px for png/svg/pdf.",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in px for png/svg/pdf.",
                    },
                },
            },
        },
        "required": ["traces"],
    },
}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    demo = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120, freq="D").repeat(3),
            "region": ["North", "South", "West"] * 120,
            "revenue": rng.normal(1000, 200, 360).round(2),
            "cost": rng.normal(700, 150, 360).round(2),
            "units": rng.integers(10, 100, 360),
        }
    )
    demo["month"] = demo["date"].dt.strftime("%Y-%m")

    print(describe_dataframe(demo), "\n")

    dashboard = {
        "title": "Sales dashboard",
        "rows": 2,
        "cols": 2,
        "subplot_titles": [
            "Monthly revenue by region",
            "Revenue share",
            "Revenue distribution",
            "Revenue vs units (monthly)",
        ],
        "traces": [
            {
                "type": "line",
                "x": "month",
                "y": "revenue",
                "agg": "sum",
                "group_by": "region",
                "markers": True,
                "row": 1,
                "col": 1,
            },
            {
                "type": "pie",
                "labels": "region",
                "values": "revenue",
                "row": 1,
                "col": 2,
            },
            {"type": "box", "x": "region", "y": "revenue", "row": 2, "col": 1},
            {
                "type": "bar",
                "x": "month",
                "y": "revenue",
                "agg": "sum",
                "name": "Revenue",
                "row": 2,
                "col": 2,
            },
            {
                "type": "line",
                "x": "month",
                "y": "units",
                "agg": "sum",
                "name": "Units",
                "secondary_y": True,
                "row": 2,
                "col": 2,
            },
        ],
        "axes": [
            {
                "row": 2,
                "col": 2,
                "y_title": "Revenue",
                "y2_title": "Units",
                "y_tickformat": "$,.0f",
            }
        ],
        "save": {
            "formats": ["html", "png", "svg", "pdf"],
            "filename": "sales_dashboard",
        },
    }
    print(
        json.dumps(
            run_plot_tool(demo, dashboard, output_dir="demo_plots"),
            indent=2,
            default=str,
        )
    )

    # Styling: fixed category colors, dashed lines, marker shapes, a gradient and bar patterns
    styled = {
        "title": "Styling options",
        "rows": 1,
        "cols": 3,
        "subplot_titles": [
            "Revenue vs cost (dashed)",
            "Units vs revenue",
            "Monthly revenue",
        ],
        "color_map": {"North": "#2563eb", "South": "#f97316", "West": "#16a34a"},
        "traces": [
            {
                "type": "line",
                "x": "month",
                "y": "revenue",
                "agg": "sum",
                "name": "Revenue",
                "color": "black",
                "line_width": 3,
                "marker_symbol": "diamond",
                "marker_size": 10,
                "row": 1,
                "col": 1,
            },
            {
                "type": "line",
                "x": "month",
                "y": "cost",
                "agg": "sum",
                "name": "Cost",
                "color": "gray",
                "line_dash": "dash",
                "row": 1,
                "col": 1,
            },
            {
                "type": "scatter",
                "x": "units",
                "y": "revenue",
                "group_by": "region",
                "symbol_by": "region",
                "marker_size": 8,
                "opacity": 0.8,
                "row": 1,
                "col": 2,
            },
            {
                "type": "bar",
                "x": "month",
                "y": "revenue",
                "agg": "sum",
                "group_by": "region",
                "pattern_by": "region",
                "row": 1,
                "col": 3,
            },
        ],
        "save": {"formats": ["html"], "filename": "styling_demo"},
    }
    print(
        json.dumps(
            run_plot_tool(demo, styled, output_dir="demo_plots"), indent=2, default=str
        )
    )

    # A deliberate mistake, to show the self-correcting error message
    bad = {"traces": [{"type": "scatter", "x": "Revenu", "y": "cost"}]}
    print(json.dumps(run_plot_tool(demo, bad, output_dir="demo_plots"), indent=2))

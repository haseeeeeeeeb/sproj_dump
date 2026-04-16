import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Union


class Analyzer:

    # ------------------------------------------------------------------ #
    #  Core                                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load()

    def _load(self) -> dict:
        with open(self.file_path, "r") as f:
            return json.load(f)

    def reset_filters(self):
        """Repopulate data from the JSON file, removing all applied filters."""
        self.data = self._load()

    def get_data(self) -> dict:
        return self.data

    def set_data(self, data: dict):
        self.data = data

    # ------------------------------------------------------------------ #
    #  Filtering                                                           #
    # ------------------------------------------------------------------ #

    def _resolve_value(self, value, reduce: str = None) -> float:
        if isinstance(value, list):
            if reduce == "mean":
                return sum(value) / len(value)
            elif reduce == "sum":
                return sum(value)
            else:
                raise ValueError(
                    "Value is an array but no reduce method given. "
                    "Use reduce='mean' or reduce='sum'."
                )
        return value

    def _in_intervals(self, value: float, intervals: list[tuple]) -> bool:
        for lo, hi in intervals:
            # Handle None values for open-ended ranges
            if lo is None and hi is None:
                return True
            elif lo is None:
                if value <= hi:
                    return True
            elif hi is None:
                if value >= lo:
                    return True
            else:
                if lo <= value <= hi:
                    return True
        return False

    def filter(self, name: str, intervals: Union[tuple, list[tuple]], reduce: str = None):
        
        if isinstance(intervals, tuple) and isinstance(intervals[0], (int, float)):
            intervals = [intervals]

        filtered = {}
        for cls_key, concepts in self.data.items():
            filtered_concepts = {}
            for concept_key, attributes in concepts.items():
                if name not in attributes:
                    continue
                value = self._resolve_value(attributes[name], reduce=reduce)
                if self._in_intervals(value, intervals):
                    filtered_concepts[concept_key] = attributes
            filtered[cls_key] = filtered_concepts

        self.data = filtered

    # ------------------------------------------------------------------ #
    #  Print                                                               #
    # ------------------------------------------------------------------ #
    def print(self):
        """Pretty-print all data with per-class meta information."""

        # ANSI color codes
        RESET   = "\033[0m"
        BOLD    = "\033[1m"
        CLASS   = "\033[38;5;39m"    # Blue  — class headings
        CONCEPT = "\033[38;5;208m"   # Orange — concept headings
        DIM     = "\033[2m"          # Dim — separators/borders

        total_concepts = sum(len(concepts) for concepts in self.data.values())

        print(f"{DIM}{'='*60}{RESET}")
        print(f"  {BOLD}Total classes{RESET} : {len(self.data)}")
        print(f"  {BOLD}Total concepts{RESET}: {total_concepts}")
        print(f"{DIM}{'='*60}{RESET}")

        for cls_key, concepts in self.data.items():
            print(f"\n  {CLASS}{BOLD}Class {cls_key}{RESET}{CLASS}  —  {len(concepts)} concept(s) remaining{RESET}")
            print(f"  {DIM}{'-'*40}{RESET}")

            for concept_key, attributes in concepts.items():
                print(f"    {CONCEPT}{BOLD}Concept {concept_key}:{RESET}")
                for attr_name, value in attributes.items():
                    if isinstance(value, list):
                        formatted = f"[{', '.join(f'{v:.4f}' for v in value)}]"
                    else:
                        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                    print(f"      {attr_name}: {formatted}")

        print(f"\n{DIM}{'='*60}{RESET}\n")

    # ------------------------------------------------------------------ #
    #  Plot                                                                #
    # ------------------------------------------------------------------ #

    def plot_concepts(
        self,
        x_name: str,
        y_name: str,
        x_reduce: str = None,
        y_reduce: str = None,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        figsize: tuple[int, int] = (10, 7),
        classes: list = None,
        view_labels: bool = False,
    ):
        """
        Scatter plot of all concepts currently in data.
        Each class gets a distinct colour; each point is optionally labelled with its concept index.

        Args:
            x_name:    Name key to use for the x-axis.
            y_name:    Name key to use for the y-axis.
            x_reduce:  'mean' or 'sum' if x values are arrays.
            y_reduce:  'mean' or 'sum' if y values are arrays.
            xlim:      (min, max) for the x-axis.
            ylim:      (min, max) for the y-axis.
            figsize:   Figure size.
            classes:   List of class keys to plot. If None, all classes are plotted.
            view_labels: If True, display concept labels on the plot. Default is False.
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Determine which classes to plot
        classes_to_plot = classes if classes is not None else list(self.data.keys())
        colors = cm.get_cmap("tab10", len(classes_to_plot))

        for cls_idx, cls_key in enumerate(classes_to_plot):
            if f"{cls_key}" not in self.data:
                continue
            concepts = self.data[f"{cls_key}"]
            color = colors(cls_idx)
            xs, ys, labels = [], [], []

            for concept_key, attributes in concepts.items():
                if x_name not in attributes or y_name not in attributes:
                    continue
                x_val = self._resolve_value(attributes[x_name], reduce=x_reduce)
                y_val = self._resolve_value(attributes[y_name], reduce=y_reduce)
                xs.append(x_val)
                ys.append(y_val)
                labels.append(concept_key)

            ax.scatter(xs, ys, color=color, label=f"Class {cls_key}", s=20, zorder=3)
            if view_labels:
                for x_val, y_val, label in zip(xs, ys, labels):
                    ax.annotate(
                        label,
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                        color=color,
                    )

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{x_name} vs {y_name}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()
        plt.show()


    def plot_plotly(
        self,
        x_name: str,
        y_name: str,
        z_name: str = None,
        x_reduce: str = None,
        y_reduce: str = None,
        z_reduce: str = None,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        zlim: tuple[float, float] = None,
        figsize: tuple[int, int] = (900, 700),
        classes: list = None,
        view_labels: bool = False,
    ):
        import plotly.graph_objects as go

        is_3d = z_name is not None
        classes_to_plot = classes if classes is not None else list(self.data.keys())

        fig = go.Figure()

        for cls_key in classes_to_plot:
            if f"{cls_key}" not in self.data:
                continue
            concepts = self.data[f"{cls_key}"]
            xs, ys, zs, labels = [], [], [], []

            for concept_key, attributes in concepts.items():
                if x_name not in attributes or y_name not in attributes:
                    continue
                if is_3d and z_name not in attributes:
                    continue

                xs.append(self._resolve_value(attributes[x_name], reduce=x_reduce))
                ys.append(self._resolve_value(attributes[y_name], reduce=y_reduce))
                labels.append(concept_key)

                if is_3d:
                    zs.append(self._resolve_value(attributes[z_name], reduce=z_reduce))

            hover_text = [f"Concept {l}" for l in labels] if view_labels else None

            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="markers+text" if view_labels else "markers",
                    marker=dict(size=4),
                    text=labels if view_labels else None,
                    textfont=dict(size=9),
                    name=f"Class {cls_key}",
                    hovertext=hover_text,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="markers+text" if view_labels else "markers",
                    marker=dict(size=6),
                    text=labels if view_labels else None,
                    textfont=dict(size=9),
                    textposition="top center",
                    name=f"Class {cls_key}",
                    hovertext=hover_text,
                ))

        title = f"{x_name} vs {y_name}" if not is_3d else f"{x_name} vs {y_name} vs {z_name}"

        axis_common = dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False)

        if is_3d:
            fig.update_layout(
                title=title,
                width=figsize[0], height=figsize[1],
                scene=dict(
                    xaxis=dict(**axis_common, title=x_name, range=list(xlim) if xlim else None),
                    yaxis=dict(**axis_common, title=y_name, range=list(ylim) if ylim else None),
                    zaxis=dict(**axis_common, title=z_name, range=list(zlim) if zlim else None),
                )
            )
        else:
            fig.update_layout(
                title=title,
                width=figsize[0], height=figsize[1],
                xaxis=dict(**axis_common, title=x_name, range=list(xlim) if xlim else None),
                yaxis=dict(**axis_common, title=y_name, range=list(ylim) if ylim else None),
                legend=dict(font=dict(size=10)),
                plot_bgcolor="white",
            )

        fig.show()


    def print_concept(self, concept_key: Union[str, int]):
        # ANSI color codes
        RESET   = "\033[0m"
        BOLD    = "\033[1m"
        CLASS   = "\033[38;5;39m"    # Blue  — class headings
        CONCEPT = "\033[38;5;208m"   # Orange — concept heading
        DIM     = "\033[2m"          # Dim — separators/borders
        FOUND   = "\033[38;5;82m"    # Green — found indicator
        MISSING = "\033[38;5;196m"   # Red   — missing indicator

        PACS_CLASSES = {
            "0": "dog",
            "1": "elephant",
            "2": "giraffe",
            "3": "guitar",
            "4": "horse",
            "5": "house",
            "6": "person",
        }

        concept_key = str(concept_key)

        print(f"{DIM}{'='*60}{RESET}")
        print(f"  {BOLD}Concept{RESET} : {CONCEPT}{BOLD}{concept_key}{RESET}")
        print(f"  {BOLD}Classes{RESET} : 0 → 6  (PACS, alphabetic order)")
        print(f"{DIM}{'='*60}{RESET}")

        found_in = 0

        for cls_idx in range(7):
            cls_key = str(cls_idx)
            class_name = PACS_CLASSES.get(cls_key, "unknown")

            print(f"\n  {CLASS}{BOLD}Class {cls_key}  —  {class_name}{RESET}")
            print(f"  {DIM}{'-'*40}{RESET}")

            concepts = self.data.get(cls_key, {})

            if concept_key not in concepts:
                print(f"    {MISSING}✗  concept {concept_key} not present{RESET}")
                continue

            found_in += 1
            attributes = concepts[concept_key]
            print(f"    {FOUND}✓  {CONCEPT}{BOLD}Concept {concept_key}:{RESET}")

            for attr_name, value in attributes.items():
                if isinstance(value, list):
                    formatted = f"[{', '.join(f'{v:.4f}' for v in value)}]"
                else:
                    formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"      {attr_name}: {formatted}")

        print(f"\n{DIM}{'='*60}{RESET}")
        print(f"  Found in {FOUND}{BOLD}{found_in}{RESET} / 7 classes")
        print(f"{DIM}{'='*60}{RESET}\n")

    def filter_across_classes(
        self,
        name: str,
        intervals: Union[tuple, list[tuple]],
        min_classes: int = 7,
        reduce: str = None):
        if isinstance(intervals, tuple) and isinstance(intervals[0], (int, float, type(None))):
            intervals = [intervals]

        # ── 1. Collect all concept keys across all classes ────────────────
        all_concept_keys = set()
        for concepts in self.data.values():
            all_concept_keys.update(concepts.keys())

        # ── 2. For each concept, count how many classes pass the condition ─
        qualifying_concepts = set()

        for concept_key in all_concept_keys:
            pass_count = 0

            for cls_key, concepts in self.data.items():
                if concept_key not in concepts:
                    continue
                attributes = concepts[concept_key]
                if name not in attributes:
                    continue
                value = self._resolve_value(attributes[name], reduce=reduce)
                if self._in_intervals(value, intervals):
                    pass_count += 1

            if pass_count >= min_classes:
                qualifying_concepts.add(concept_key)

        # ── 3. Rebuild self.data keeping only qualifying concepts ──────────
        filtered = {}
        for cls_key, concepts in self.data.items():
            filtered[cls_key] = {
                concept_key: attributes
                for concept_key, attributes in concepts.items()
                if concept_key in qualifying_concepts
            }

        self.data = filtered
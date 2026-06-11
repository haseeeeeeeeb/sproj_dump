import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Union
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import torch


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
        logx: bool = False,
        logy: bool = False,
        
    ):
        """
        Scatter plot of all concepts currently in data.
        Each class gets a distinct colour; each point is optionally labelled with its concept index.

        Args:
            x_name:      Name key to use for the x-axis.
            y_name:      Name key to use for the y-axis.
            x_reduce:    'mean' or 'sum' if x values are arrays.
            y_reduce:    'mean' or 'sum' if y values are arrays.
            xlim:        (min, max) for the x-axis.
            ylim:        (min, max) for the y-axis.
            figsize:     Figure size.
            classes:     List of class keys to plot. If None, all classes are plotted.
            view_labels: If True, display concept labels on the plot. Default is False.
            logx:        If True, plot the x-axis on a logarithmic scale.
            logy:        If True, plot the y-axis on a logarithmic scale.
        """
        fig, ax = plt.subplots(figsize=figsize)
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

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

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



    # ------------------------------------------------------------------ #
    def plot_concepts_with_dict(
        self,
        x_name: str,
        y_name: str,
        data: dict,
        x_reduce: str = None,
        y_reduce: str = None,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        figsize: tuple[int, int] = (10, 7),
        classes: list = None,
        view_labels: bool = False,
        logx: bool = False,
        logy: bool = False,
        fit_line: bool = False,  # <-- NEW PARAMETER
    ):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = cm.get_cmap("tab10", len(data.keys()))

        # Distinct markers to differentiate between datasets
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]

        for ds_idx, (ds_name, data_dict) in enumerate(data.items()):
            marker = markers[ds_idx % len(markers)]

            for cls_idx, cls_key in enumerate(classes):
                if f"{cls_key}" not in data_dict:
                    continue
                
                concepts = data_dict[f"{cls_key}"]
                color = colors(cls_idx + ds_idx)
                xs, ys, labels = [], [], []

                for concept_key, attributes in concepts.items():
                    if x_name not in attributes or y_name not in attributes:
                        continue
                    
                    x_val = self._resolve_value(attributes[x_name], reduce=x_reduce)
                    y_val = self._resolve_value(attributes[y_name], reduce=y_reduce)
                    
                    xs.append(x_val)
                    ys.append(y_val)
                    labels.append(concept_key)

                # Append dataset name to label to differentiate in legend
                legend_label = f"{ds_name} - Class {cls_key}"
                
                ax.scatter(
                    xs, ys, 
                    color=color, 
                    marker=marker, 
                    label=legend_label, 
                    s=10, 
                    zorder=3
                )

                # --- NEW FIT LINE LOGIC ---
                if fit_line and len(xs) > 1:
                    import numpy as np
                    xs_arr = np.array(xs)
                    ys_arr = np.array(ys)
                    
                    if logx and logy:
                        # 1. Filter out zeros/negatives (log of <= 0 is undefined)
                        valid = (xs_arr > 0) & (ys_arr > 0)
                        
                        if np.sum(valid) > 1:
                            x_val = xs_arr[valid]
                            y_val = ys_arr[valid]
                            
                            # 2. Calculate fit on the log10 of the data
                            m, b = np.polyfit(np.log10(x_val), np.log10(y_val), 1)
                            
                            # 3. Create the fit line coordinates
                            x_fit = np.array([min(x_val), max(x_val)])
                            
                            # y = 10^(m * log10(x) + b)
                            y_fit = 10**(m * np.log10(x_fit) + b)
                            
                            ax.plot(x_fit, y_fit, color=color, linestyle="--", alpha=0.8, zorder=2)
                    else:
                        # Standard linear fit if not on log scales
                        m, b = np.polyfit(xs_arr, ys_arr, 1)
                        x_fit = np.array([min(xs_arr), max(xs_arr)])
                        y_fit = m * x_fit + b
                        ax.plot(x_fit, y_fit, color=color, linestyle="--", alpha=0.8, zorder=2)
                # --------------------------
                
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

        # Move legend outside if it gets too large from multiple datasets
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()
        plt.show()


    def get_mask(self):
            """
            Returns a boolean PyTorch tensor of dimensions (classes x total_concepts).
            Everything is handled within PyTorch.
            """
            # 1. Load original data to define the full grid dimensions
            original_data = self._load()
            
            # 2. Identify and sort keys numerically to maintain index consistency
            all_class_keys = sorted(original_data.keys(), key=lambda x: int(x))
            
            all_concept_keys = set()
            for concepts in original_data.values():
                all_concept_keys.update(concepts.keys())
            all_concept_keys = sorted(list(all_concept_keys), key=lambda x: int(x))
            
            nb_classes = len(all_class_keys)
            nb_concepts = len(all_concept_keys)
            
            # 3. Initialize a PyTorch boolean tensor filled with False
            mask = torch.zeros((nb_classes, nb_concepts), dtype=torch.bool)
            
            # 4. Map keys to their respective index positions
            class_to_idx = {key: i for i, key in enumerate(all_class_keys)}
            concept_to_idx = {key: i for i, key in enumerate(all_concept_keys)}
            
            # 5. Populate the mask based on the current (filtered) state of self.data
            for cls_key, concepts in self.data.items():
                if cls_key in class_to_idx:
                    cls_idx = class_to_idx[cls_key]
                    for concept_key in concepts.keys():
                        if concept_key in concept_to_idx:
                            concept_idx = concept_to_idx[concept_key]
                            mask[cls_idx, concept_idx] = True

            ## invert mask (True for concepts not present, false for those present)
            mask = ~mask                
            return mask

    def filter_by_cluster(self, names: Union[str, list[str]], n_clusters: int, keep_clusters: list[int] = None, reduce: str = "mean"):
        
        if isinstance(names, str):
            names = [names]

        new_data = {}

        # ANSI colors for the display
        BOLD = "\033[1m"
        RESET = "\033[0m"
        DIM = "\033[2m"
        CLASS_CLR = "\033[38;5;39m"

        print(f"\n{BOLD}Per-Class Cluster Analysis (k={n_clusters}){RESET}")

        for cls_key, concepts in self.data.items():
            if not concepts:
                new_data[cls_key] = {}
                continue

            # 1. Extract features for this specific class
            concept_keys = []
            features = []
            for c_key, attrs in concepts.items():
                try:
                    vec = [self._resolve_value(attrs[n], reduce=reduce) for n in names]
                    features.append(vec)
                    concept_keys.append(c_key)
                except KeyError:
                    continue

            X = np.array(features)

            # Safety check: K-Means needs at least as many points as clusters
            current_k = min(n_clusters, len(X))
            if current_k < 1:
                new_data[cls_key] = {}
                continue

            # 2. Run KMeans locally
            kmeans = KMeans(n_clusters=current_k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_

            # 3. Print report for this class
            print(f"\n  {CLASS_CLR}{BOLD}Class {cls_key}{RESET} ({len(X)} concepts)")
            for i, center in enumerate(centroids):
                formatted_center = ", ".join([f"{names[j]}: {center[j]:.4f}" for j in range(len(names))])
                count = np.sum(labels == i)
                print(f"    Cluster {i}: ({formatted_center}) {DIM}[{count} concepts]{RESET}")

            # 4. Apply filter
            if keep_clusters is not None:
                filtered_concepts = {}
                for idx, label in enumerate(labels):
                    if label in keep_clusters:
                        c_key = concept_keys[idx]
                        filtered_concepts[c_key] = concepts[c_key]
                new_data[cls_key] = filtered_concepts
            else:
                # If no keep_clusters provided, we don't change the data
                new_data[cls_key] = concepts

        if keep_clusters is not None:
            self.data = new_data
            print(f"\n{BOLD}Filtering complete.{RESET} Kept clusters: {keep_clusters} across all classes.")
        else:
            print(f"\n{DIM}No clusters specified to keep. Data remains unchanged.{RESET}")
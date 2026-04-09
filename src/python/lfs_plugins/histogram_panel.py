# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floating retained panel for Gaussian histogram analysis."""

from __future__ import annotations

import math
from typing import Iterable

import lichtfeld as lf

from .histogram_support import METRICS, METRIC_BY_ID, histogram_mode_available, histogram_tr
from .types import Panel
from .ui.state import AppState


BIN_COUNT = 56


def _tr(key: str, fallback: str) -> str:
    return histogram_tr(key, fallback)


def _trf(key: str, fallback: str, **kwargs) -> str:
    return _tr(key, fallback).format(**kwargs)


class HistogramPanel(Panel):
    id = "lfs.histogram"
    label = "Histogram"
    space = lf.ui.PanelSpace.FLOATING
    order = 97
    template = "rmlui/histogram_panel.rml"
    size = (860, 660)
    update_interval_ms = 250

    def __init__(self):
        self._doc = None
        self._chart_el = None
        self._close_btn = None
        self._handle = None

        self._metric_id = METRICS[0].id
        self._log_scale_enabled = False
        self._scene_generation = -1
        self._history_generation = -1
        self._last_lang = ""
        self._trainer_state = ""
        self._show_chart = False
        self._empty_title = _tr("histogram.empty.unavailable.title", "Histogram unavailable")
        self._empty_message = _tr(
            "histogram.empty.unavailable.message",
            "Switch to a view or edit scene with Gaussian splats to inspect a distribution.",
        )
        self._sample_count = "--"
        self._range_text = "--"
        self._mean_text = "--"
        self._median_text = "--"
        self._p95_text = "--"
        self._peak_text = "--"
        self._axis_min = "--"
        self._axis_max = "--"
        self._summary_text = ""

        self._selection_bin_indices: lf.Tensor | None = None
        self._hist_counts: list[int] | None = None
        self._hist_prefix_counts: list[int] | None = None
        self._hist_edges: list[float] | None = None
        self._selection_owned = False
        self._pending_selection_commit = False

        self._dragging_mark = False
        self._marked_bin_start: int | None = None
        self._marked_bin_end: int | None = None
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"

    @classmethod
    def poll(cls, context):
        return histogram_mode_available(context)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("histogram_panel")
        if model is None:
            return

        model.bind_func("panel_label", lambda: _tr("window.histogram", "Histogram"))
        model.bind_func("analysis_label", lambda: _tr("histogram.title_eyebrow", "Gaussian Analysis"))
        model.bind_func("field_label", lambda: _tr("histogram.field", "Field"))
        model.bind_func("log_scale_label", lambda: _tr("histogram.log_scale", "Log Scale"))
        model.bind_func("samples_label", lambda: _tr("histogram.samples", "Samples"))
        model.bind_func("range_label", lambda: _tr("histogram.range", "Range"))
        model.bind_func("mean_label", lambda: _tr("histogram.mean", "Mean"))
        model.bind_func("median_label", lambda: _tr("histogram.median", "Median"))
        model.bind_func("p95_label", lambda: _tr("histogram.p95", "P95"))
        model.bind_func("peak_bin_label", lambda: _tr("histogram.peak_bin", "Peak Bin"))
        model.bind_func("bin_count_text", lambda: _trf("histogram.bin_count", "{count} bins", count=BIN_COUNT))
        model.bind_func("marked_label", lambda: _tr("histogram.marked", "Marked"))
        model.bind_func("count_label", lambda: _tr("histogram.count", "Count"))
        model.bind_func("undo_tooltip", self._undo_tooltip)
        model.bind_func("redo_tooltip", self._redo_tooltip)
        model.bind_func("clear_label", lambda: _tr("histogram.clear", "Clear"))
        model.bind_func("delete_label", lambda: _tr("histogram.delete", "Delete"))
        model.bind_func("metric_title", lambda: METRIC_BY_ID[self._metric_id].label())
        model.bind_func("metric_description", lambda: METRIC_BY_ID[self._metric_id].description())
        model.bind_func("show_chart", lambda: self._show_chart)
        model.bind_func("show_empty_state", lambda: not self._show_chart)
        model.bind_func("empty_title", lambda: self._empty_title)
        model.bind_func("empty_message", lambda: self._empty_message)
        model.bind_func("sample_count", lambda: self._sample_count)
        model.bind_func("range_text", lambda: self._range_text)
        model.bind_func("mean_text", lambda: self._mean_text)
        model.bind_func("median_text", lambda: self._median_text)
        model.bind_func("p95_text", lambda: self._p95_text)
        model.bind_func("peak_text", lambda: self._peak_text)
        model.bind_func("axis_min", lambda: self._axis_min)
        model.bind_func("axis_max", lambda: self._axis_max)
        model.bind_func("summary_text", lambda: self._summary_text)
        model.bind_func("show_selection_overlay", self._has_marked_range)
        model.bind_func("selection_left_style", lambda: self._selection_left_style)
        model.bind_func("selection_width_style", lambda: self._selection_width_style)
        model.bind_func("marked_range_text", lambda: self._marked_range_text)
        model.bind_func("marked_count_text", lambda: self._marked_count_text)
        model.bind_func("status_hint", lambda: self._status_hint)
        model.bind_func("undo_enabled", self._can_undo)
        model.bind_func("redo_enabled", self._can_redo)
        model.bind_func("clear_enabled", self._has_marked_range)
        model.bind_func("delete_enabled", lambda: self._has_marked_range() and self._marked_count > 0)
        model.bind("metric_id", lambda: self._metric_id, self._set_metric_id)
        model.bind("log_scale_enabled", lambda: self._log_scale_enabled, self._set_log_scale_enabled)
        model.bind_event("undo_history", self._on_undo_history)
        model.bind_event("redo_history", self._on_redo_history)
        model.bind_event("clear_mark", self._on_clear_mark)
        model.bind_event("delete_marked", self._on_delete_marked)
        model.bind_record_list("metric_options")
        model.bind_record_list("bins")

        self._handle = model.get_handle()
        self._rebuild_metric_options()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._chart_el = doc.get_element_by_id("histogram-bars")
        self._close_btn = doc.get_element_by_id("close-btn")
        if self._chart_el:
            self._chart_el.add_event_listener("mousedown", self._on_chart_mousedown)
        if self._close_btn:
            self._close_btn.add_event_listener("click", self._on_close_click)
        doc.add_event_listener("mousemove", self._on_document_mousemove)
        doc.add_event_listener("mouseup", self._on_document_mouseup)

        self._scene_generation = -1
        self._history_generation = self._history_generation_value()
        self._last_lang = lf.ui.get_current_language()
        self._trainer_state = ""
        self._rebuild_metric_options()
        self._refresh()

    def on_update(self, doc):
        del doc

        if not histogram_mode_available():
            lf.ui.set_panel_enabled(self.id, False)
            return False

        scene_generation = lf.get_scene_generation()
        history_generation = self._history_generation_value()
        current_lang = lf.ui.get_current_language()
        trainer_state = AppState.trainer_state.value
        scene_changed = scene_generation != self._scene_generation
        history_changed = history_generation != self._history_generation
        if (scene_generation == self._scene_generation and
                history_generation == self._history_generation and
                trainer_state == self._trainer_state and
                current_lang == self._last_lang):
            return False

        if scene_changed or history_changed:
            if self._pending_selection_commit:
                self._pending_selection_commit = False
            else:
                self._selection_owned = False

        self._scene_generation = scene_generation
        self._history_generation = history_generation
        self._last_lang = current_lang
        self._trainer_state = trainer_state
        self._rebuild_metric_options()
        self._refresh()
        return True

    def on_scene_changed(self, doc):
        del doc
        self._scene_generation = -1

    def on_unmount(self, doc):
        self._clear_owned_scene_selection()
        self._doc = None
        self._chart_el = None
        self._close_btn = None
        self._handle = None
        doc.remove_data_model("histogram_panel")

    def _set_metric_id(self, value):
        metric_id = str(value)
        if metric_id not in METRIC_BY_ID or metric_id == self._metric_id:
            return
        self._clear_marked_range(clear_scene=True)
        self._metric_id = metric_id
        self._rebuild_metric_options()
        self._refresh()

    def _set_log_scale_enabled(self, value):
        enabled = self._coerce_bool(value)
        if enabled == self._log_scale_enabled:
            return
        self._log_scale_enabled = enabled
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _rebuild_metric_options(self):
        if not self._handle:
            return
        self._handle.update_record_list(
            "metric_options",
            [{"value": metric.id, "label": metric.label()} for metric in METRICS],
        )

    def _refresh(self):
        if not self._handle:
            return

        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            self._set_empty(
                _tr("histogram.empty.no_scene.title", "No scene loaded"),
                _tr("histogram.empty.no_scene.message", "Load a Gaussian scene, then reopen the histogram panel."),
            )
            return

        model = scene.combined_model()
        if model is None or int(getattr(model, "num_points", 0) or 0) <= 0:
            self._set_empty(
                _tr("histogram.empty.no_visible_gaussians.title", "No visible Gaussians"),
                _tr(
                    "histogram.empty.no_visible_gaussians.message",
                    "The histogram only works when the active scene exposes a visible Gaussian model.",
                ),
            )
            return

        values = self._extract_metric_values(scene, model)
        if values is None:
            self._set_empty(
                _tr("histogram.empty.metric_unavailable.title", "Metric unavailable"),
                _tr(
                    "histogram.empty.metric_unavailable.message",
                    "The selected metric could not be read from the combined Gaussian model.",
                ),
            )
            return

        visible_mask = self._extract_visible_mask(model, values)
        finite_mask = values.isfinite()
        if visible_mask is not None and visible_mask.shape == values.shape:
            finite_mask = finite_mask & visible_mask

        if not self._any_true(finite_mask):
            self._set_empty(
                _tr("histogram.empty.no_visible_values.title", "No visible values"),
                _tr(
                    "histogram.empty.no_visible_values.message",
                    "The selected metric does not contain any visible finite samples to visualize.",
                ),
            )
            return

        valid_values = values[finite_mask]
        finite_values = valid_values.contiguous().cpu().to("float32")
        metric = METRIC_BY_ID[self._metric_id]
        histogram_min, histogram_max = self._histogram_bounds(finite_values)
        selection_bin_indices = self._build_selection_bin_indices(values, finite_mask, histogram_min, histogram_max)
        valid_bin_indices = self._bin_indices_for_values(valid_values, histogram_min, histogram_max)
        counts, edges = self._build_histogram(valid_bin_indices, int(finite_values.shape[0]), histogram_min, histogram_max)
        peak_count = max(counts, default=0)
        sorted_values, _ = finite_values.sort(0, False)

        self._selection_bin_indices = selection_bin_indices
        self._hist_counts = counts
        self._hist_prefix_counts = self._prefix_counts(counts)
        self._hist_edges = edges

        self._show_chart = True
        self._sample_count = f"{int(finite_values.shape[0]):,}"
        self._range_text = self._format_range_text(finite_values.min_scalar(), finite_values.max_scalar())
        self._mean_text = self._format_value(finite_values.mean_scalar())
        self._median_text = self._format_value(self._percentile_from_sorted(sorted_values, 50.0))
        self._p95_text = self._format_value(self._percentile_from_sorted(sorted_values, 95.0))
        self._peak_text = f"{peak_count:,}"
        self._axis_min = self._format_value(edges[0])
        self._axis_max = self._format_value(edges[-1])
        self._summary_text = _trf(
            "histogram.summary",
            "{metric} distribution across {count} Gaussians",
            metric=metric.label(),
            count=f"{int(finite_values.shape[0]):,}",
        )

        if self._has_marked_range():
            self._sync_marked_range(apply_scene=False)
        else:
            self._reset_marked_state(clear_scene=False)

        self._update_bin_records()
        self._handle.dirty_all()

    def _set_empty(self, title: str, message: str):
        self._show_chart = False
        self._empty_title = title
        self._empty_message = message
        self._sample_count = "--"
        self._range_text = "--"
        self._mean_text = "--"
        self._median_text = "--"
        self._p95_text = "--"
        self._peak_text = "--"
        self._axis_min = "--"
        self._axis_max = "--"
        self._summary_text = ""
        self._selection_bin_indices = None
        self._hist_counts = None
        self._hist_prefix_counts = None
        self._hist_edges = None
        self._reset_marked_state(clear_scene=True)
        if self._handle:
            self._handle.update_record_list("bins", [])
            self._handle.dirty_all()

    def _build_bin_records(self, counts: list[int], edges: list[float]) -> Iterable[dict[str, object]]:
        if self._log_scale_enabled:
            display_counts = [math.log1p(float(count)) for count in counts]
        else:
            display_counts = [float(count) for count in counts]

        peak = max(max(display_counts, default=0.0), 1.0)
        marked_lo, marked_hi = self._marked_bounds()
        for index, count in enumerate(counts):
            ratio = display_counts[index] / peak
            height_pct = 0.0 if count <= 0 else max(3.0, ratio * 100.0)
            alpha = 0.16 if count <= 0 else (0.34 + ratio * 0.66)
            left = self._format_value(edges[index])
            right = self._format_value(edges[index + 1])
            yield {
                "height_style": f"{height_pct:.2f}%",
                "opacity_style": f"{alpha:.3f}",
                "tooltip": _trf(
                    "histogram.bin_tooltip",
                    "Bin {index}: {left} to {right} | {count} Gaussians",
                    index=index + 1,
                    left=left,
                    right=right,
                    count=f"{int(count):,}",
                ),
                "selected": marked_lo is not None and marked_lo <= index <= marked_hi,
            }

    def _update_bin_records(self):
        if self._handle is None or self._hist_counts is None or self._hist_edges is None:
            return
        self._handle.update_record_list(
            "bins",
            list(self._build_bin_records(self._hist_counts, self._hist_edges)),
        )

    def _extract_metric_values(self, scene, model) -> lf.Tensor | None:
        try:
            if self._metric_id == "opacity":
                return self._float_tensor(model.get_opacity()).reshape([-1])

            if self._metric_id in {"position_x", "position_y", "position_z", "distance"}:
                world_means = self._extract_world_space_means(scene, model)
                if self._metric_id == "position_x":
                    return world_means[:, 0]
                if self._metric_id == "position_y":
                    return world_means[:, 1]
                if self._metric_id == "position_z":
                    return world_means[:, 2]
                return self._distance_from_positions(scene, world_means)

            scaling = self._float_tensor(model.get_scaling()).reshape([-1, 3])
            if self._metric_id == "scale_x":
                return scaling[:, 0]
            if self._metric_id == "scale_y":
                return scaling[:, 1]
            if self._metric_id == "scale_z":
                return scaling[:, 2]
            if self._metric_id == "scale_max":
                return scaling.max(1).reshape([-1])
            if self._metric_id == "volume":
                return scaling.prod(1).reshape([-1]) * (4.0 * math.pi / 3.0)
            if self._metric_id == "anisotropy":
                scale_min = scaling.min(1).reshape([-1])
                scale_max = scaling.max(1).reshape([-1])
                return scale_max / (scale_min + 1e-12)
            return None
        except Exception:
            return None

    def _extract_world_space_means(self, scene, model) -> lf.Tensor:
        means = self._float_tensor(model.get_means()).reshape([-1, 3])
        if means.numel == 0:
            return lf.Tensor.zeros([0, 3], dtype="float32", device=self._device_string(means))

        world_means = self._world_space_means(scene, means)
        return means if world_means is None else world_means

    def _world_space_means(self, scene, means: lf.Tensor) -> lf.Tensor | None:
        nodes = self._visible_splat_nodes(scene)
        if not nodes:
            return None

        world_means = means.clone()
        offset = 0

        for node in nodes:
            count = int(getattr(node, "gaussian_count", 0) or 0)
            if count <= 0:
                continue

            next_offset = offset + count
            if next_offset > world_means.shape[0]:
                return None

            matrix = lf.mat4([list(row) for row in node.world_transform]).to("float32")
            matrix = self._to_device(matrix, self._device_string(world_means))
            rotation = matrix[:3, :3].transpose(0, 1)
            translation = matrix[:3, 3].unsqueeze(0).expand([count, 3])
            world_means[offset:next_offset] = world_means[offset:next_offset].matmul(rotation) + translation
            offset = next_offset

        if offset != world_means.shape[0]:
            return None
        return world_means

    def _distance_from_positions(self, scene, positions: lf.Tensor) -> lf.Tensor:
        finite_rows = positions.isfinite().all(1).reshape([-1])
        if not self._any_true(finite_rows):
            return self._nan_tensor(int(positions.shape[0]), self._device_string(positions))

        center = self._resolve_scene_center(scene, positions, finite_rows)
        distances = self._nan_tensor(int(positions.shape[0]), self._device_string(positions))
        centered = positions[finite_rows] - center.unsqueeze(0)
        distances[finite_rows] = centered.square().sum(1).sqrt().reshape([-1])
        return distances

    def _visible_splat_nodes(self, scene) -> list:
        try:
            nodes = list(scene.get_nodes())
        except Exception:
            return []

        node_by_id = {}
        for node in nodes:
            try:
                node_by_id[int(node.id)] = node
            except Exception:
                continue

        visible_cache: dict[int, bool] = {}
        splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
        if splat_type is None:
            splat_type = getattr(getattr(getattr(lf, "scene", None), "NodeType", None), "SPLAT", None)

        def is_effectively_visible(node) -> bool:
            node_id = int(node.id)
            if node_id in visible_cache:
                return visible_cache[node_id]

            visible = bool(getattr(node, "visible", True))
            parent_id = int(getattr(node, "parent_id", -1))
            if visible and parent_id >= 0:
                parent = node_by_id.get(parent_id)
                if parent is not None:
                    visible = is_effectively_visible(parent)

            visible_cache[node_id] = visible
            return visible

        splat_nodes = []
        for node in nodes:
            try:
                if splat_type is not None and getattr(node, "type", None) != splat_type:
                    continue
                if int(getattr(node, "gaussian_count", 0) or 0) <= 0:
                    continue
                if is_effectively_visible(node):
                    splat_nodes.append(node)
            except Exception:
                continue
        return splat_nodes

    def _resolve_scene_center(self, scene, means: lf.Tensor, finite_rows: lf.Tensor) -> lf.Tensor:
        try:
            center = getattr(scene, "scene_center", None)
            if center is not None and int(getattr(center, "ndim", 0) or 0) == 1 and tuple(center.shape) == (3,):
                center = self._float_tensor(center).reshape([-1])
                center = self._to_device(center, self._device_string(means))
                if center.isfinite().all().bool_():
                    return center
        except Exception:
            pass

        finite_means = means[finite_rows]
        if finite_means.numel == 0:
            return lf.Tensor.zeros([3], dtype="float32", device=self._device_string(means))
        return finite_means.mean(0).reshape([-1]).to("float32")

    @staticmethod
    def _float_tensor(tensor) -> lf.Tensor:
        return tensor.contiguous().to("float32")

    @staticmethod
    def _device_string(tensor: lf.Tensor) -> str:
        return "cuda" if bool(getattr(tensor, "is_cuda", False)) else "cpu"

    @staticmethod
    def _to_device(tensor: lf.Tensor, device: str) -> lf.Tensor:
        if device == "cuda":
            return tensor if tensor.is_cuda else tensor.cuda()
        return tensor.cpu() if tensor.is_cuda else tensor

    @staticmethod
    def _any_true(mask: lf.Tensor) -> bool:
        return bool(mask.count_nonzero())

    @staticmethod
    def _nan_tensor(size: int, device: str) -> lf.Tensor:
        return lf.Tensor.full([size], float("nan"), dtype="float32", device=device)

    def _extract_visible_mask(self, model, values: lf.Tensor) -> lf.Tensor | None:
        try:
            if bool(model.has_deleted_mask()):
                deleted = getattr(model, "deleted", None)
                if deleted is None or int(getattr(deleted, "ndim", 0) or 0) != 1:
                    return None
                if int(deleted.shape[0]) != int(values.shape[0]):
                    return None
                deleted = deleted.contiguous().reshape([-1]).to("bool")
                deleted = self._to_device(deleted, self._device_string(values))
                return ~deleted
        except Exception:
            pass
        return None

    def _histogram_bounds(self, values: lf.Tensor) -> tuple[float, float]:
        if self._metric_id == "opacity":
            return 0.0, 1.0
        if self._metric_id == "anisotropy":
            lo = 1.0
            hi = values.max_scalar()
            if not math.isfinite(hi):
                return lo, lo + 1.0
            if hi < lo:
                hi = lo
            if math.isclose(hi, lo, rel_tol=1e-6, abs_tol=1e-9):
                return lo, lo + 1e-3
            return lo, hi

        lo = values.min_scalar()
        hi = values.max_scalar()
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 0.0, 1.0

        if math.isclose(lo, hi, rel_tol=1e-6, abs_tol=1e-9):
            padding = max(abs(lo) * 0.05, 1e-3)
            return lo - padding, hi + padding

        return lo, hi

    def _build_histogram(
        self,
        bin_indices: lf.Tensor,
        value_count: int,
        histogram_min: float,
        histogram_max: float,
    ) -> tuple[list[int], list[float]]:
        edges = [
            histogram_min + (histogram_max - histogram_min) * (index / BIN_COUNT)
            for index in range(BIN_COUNT + 1)
        ]

        span = histogram_max - histogram_min
        if not math.isfinite(span) or span <= 0.0:
            counts = [0] * BIN_COUNT
            if value_count > 0:
                counts[-1] = value_count
            return counts, edges

        device = self._device_string(bin_indices)
        counts_tensor = lf.Tensor.zeros([BIN_COUNT], dtype="int32", device=device)
        if value_count > 0:
            ones = lf.Tensor.ones([value_count], dtype="int32", device=device)
            counts_tensor.index_add_(0, bin_indices.contiguous().to("int32"), ones)
        counts = counts_tensor.cpu().tolist() if counts_tensor.is_cuda else counts_tensor.tolist()
        counts = [int(count) for count in counts]
        return counts, edges

    @staticmethod
    def _bin_indices_for_values(values: lf.Tensor, histogram_min: float, histogram_max: float) -> lf.Tensor:
        value_count = int(values.shape[0]) if values.ndim > 0 else int(values.numel)
        device = HistogramPanel._device_string(values)
        if value_count <= 0:
            return lf.Tensor.zeros([0], dtype="int32", device=device)

        span = histogram_max - histogram_min
        if not math.isfinite(span) or span <= 0.0:
            return lf.Tensor.full([value_count], BIN_COUNT - 1, dtype="int32", device=device)

        return (
            (((values - histogram_min) / span) * BIN_COUNT)
            .floor()
            .clamp(0.0, float(BIN_COUNT - 1))
            .reshape([-1])
            .to("int32")
        )

    def _build_selection_bin_indices(
        self,
        values: lf.Tensor,
        finite_mask: lf.Tensor,
        histogram_min: float,
        histogram_max: float,
    ) -> lf.Tensor:
        value_count = int(values.shape[0]) if values.ndim > 0 else int(values.numel)
        device = self._device_string(values)
        selection_bin_indices = lf.Tensor.full([value_count], -1, dtype="int32", device=device)
        if value_count <= 0 or not self._any_true(finite_mask):
            return selection_bin_indices

        selection_bin_indices[finite_mask] = self._bin_indices_for_values(
            values[finite_mask],
            histogram_min,
            histogram_max,
        )
        return selection_bin_indices

    @staticmethod
    def _prefix_counts(counts: list[int]) -> list[int]:
        prefix = [0]
        running = 0
        for count in counts:
            running += int(count)
            prefix.append(running)
        return prefix

    def _marked_count_for_bins(self, lo: int, hi: int) -> int:
        if self._hist_prefix_counts is None:
            return 0
        return int(self._hist_prefix_counts[hi + 1] - self._hist_prefix_counts[lo])

    @staticmethod
    def _percentile_from_sorted(sorted_values: lf.Tensor, percentile: float) -> float:
        count = int(sorted_values.shape[0])
        if count <= 0:
            return 0.0
        if count == 1:
            return sorted_values[0].item()

        position = (count - 1) * max(0.0, min(percentile, 100.0)) / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower].item()

        weight = position - lower
        lower_value = sorted_values[lower].item()
        upper_value = sorted_values[upper].item()
        return lower_value + (upper_value - lower_value) * weight

    def _on_chart_mousedown(self, event):
        if not self._show_chart or self._chart_el is None or self._hist_edges is None:
            return
        if int(event.get_parameter("button", "0")) != 0:
            return

        bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
        self._dragging_mark = True
        self._marked_bin_start = bin_index
        self._marked_bin_end = bin_index
        self._sync_marked_range(apply_scene=False)
        event.stop_propagation()

    def _on_document_mousemove(self, event):
        if not self._dragging_mark or self._chart_el is None:
            return

        bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
        if bin_index == self._marked_bin_end:
            return
        self._marked_bin_end = bin_index
        self._sync_marked_range(apply_scene=False)
        event.stop_propagation()

    def _on_document_mouseup(self, event):
        if not self._dragging_mark:
            return

        self._dragging_mark = False
        self._sync_marked_range(apply_scene=True)
        event.stop_propagation()

    def _event_mouse_x(self, event) -> float:
        try:
            return float(event.get_parameter("mouse_x", "0"))
        except Exception:
            return 0.0

    def _bin_index_for_mouse_x(self, mouse_x: float) -> int:
        if self._chart_el is None:
            return 0
        left = float(self._chart_el.absolute_left)
        width = max(float(self._chart_el.absolute_width), 1.0)
        norm = min(1.0, max(0.0, (mouse_x - left) / width))
        return min(BIN_COUNT - 1, max(0, int(math.floor(norm * BIN_COUNT))))

    def _marked_bounds(self) -> tuple[int | None, int | None]:
        if self._marked_bin_start is None or self._marked_bin_end is None:
            return None, None
        lo = max(0, min(self._marked_bin_start, self._marked_bin_end))
        hi = min(BIN_COUNT - 1, max(self._marked_bin_start, self._marked_bin_end))
        return lo, hi

    def _has_marked_range(self) -> bool:
        lo, hi = self._marked_bounds()
        return lo is not None and hi is not None

    def _sync_marked_range(self, apply_scene: bool):
        if self._hist_counts is None or self._hist_edges is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        lo, hi = self._marked_bounds()
        if lo is None or hi is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        left_ratio = lo / BIN_COUNT
        width_ratio = (hi + 1) / BIN_COUNT - left_ratio
        self._selection_left_style = f"{left_ratio * 100.0:.2f}%"
        self._selection_width_style = f"{max(width_ratio * 100.0, 1.0):.2f}%"

        range_min = float(self._hist_edges[lo])
        range_max = float(self._hist_edges[hi + 1])
        self._marked_count = self._marked_count_for_bins(lo, hi)
        self._marked_range_text = self._format_range_text(range_min, range_max)
        self._marked_count_text = _trf(
            "histogram.gaussian_count",
            "{count} Gaussians",
            count=f"{self._marked_count:,}",
        )
        self._status_hint = _tr(
            "histogram.status_selection",
            "Marked range becomes the active Gaussian selection.",
        )

        self._update_bin_records()
        if apply_scene:
            self._apply_scene_selection(lo, hi)
        if self._handle:
            self._handle.dirty_all()

    def _reset_marked_state(self, clear_scene: bool):
        self._marked_bin_start = None
        self._marked_bin_end = None
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"
        if clear_scene:
            self._clear_owned_scene_selection()
        else:
            self._selection_owned = False
            self._pending_selection_commit = False

    def _clear_marked_range(self, clear_scene: bool):
        self._reset_marked_state(clear_scene=clear_scene)
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _selection_mask_for_bins(self, lo: int, hi: int) -> lf.Tensor | None:
        if self._selection_bin_indices is None:
            return None

        mask = self._selection_bin_indices >= lo
        mask = mask & (self._selection_bin_indices <= hi)
        return mask

    def _apply_scene_selection(self, lo: int, hi: int):
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            self._selection_owned = False
            self._pending_selection_commit = False
            return

        mask = self._selection_mask_for_bins(lo, hi)
        if mask is None or not self._any_true(mask):
            scene.clear_selection()
            self._selection_owned = False
            self._pending_selection_commit = False
            return

        try:
            scene.set_selection_mask(mask.contiguous())
            self._selection_owned = True
            self._pending_selection_commit = True
        except Exception:
            self._selection_owned = False
            self._pending_selection_commit = False

    def _clear_owned_scene_selection(self):
        scene = lf.get_scene()
        if scene is not None and scene.is_valid() and self._selection_owned:
            scene.clear_selection()
        self._selection_owned = False
        self._pending_selection_commit = False

    @staticmethod
    def _history_generation_value() -> int:
        try:
            return int(lf.undo.generation())
        except Exception:
            return -1

    @staticmethod
    def _can_undo() -> bool:
        try:
            return bool(lf.undo.can_undo())
        except Exception:
            return False

    @staticmethod
    def _can_redo() -> bool:
        try:
            return bool(lf.undo.can_redo())
        except Exception:
            return False

    @staticmethod
    def _history_name(get_name) -> str:
        try:
            return str(get_name() or "").strip()
        except Exception:
            return ""

    def _undo_tooltip(self) -> str:
        name = self._history_name(lf.undo.get_undo_name) if self._can_undo() else ""
        if name:
            return _trf("histogram.undo_named", "Undo: {name}", name=name)
        return _tr("histogram.undo", "Undo")

    def _redo_tooltip(self) -> str:
        name = self._history_name(lf.undo.get_redo_name) if self._can_redo() else ""
        if name:
            return _trf("histogram.redo_named", "Redo: {name}", name=name)
        return _tr("histogram.redo", "Redo")

    def _on_clear_mark(self, _handle, _event, _args):
        self._clear_marked_range(clear_scene=True)

    def _on_close_click(self, event):
        self._clear_owned_scene_selection()
        try:
            lf.selection.clear_preview()
        except Exception:
            pass
        event.stop_propagation()
        lf.ui.set_panel_enabled(self.id, False)

    def _on_undo_history(self, _handle, _event, _args):
        try:
            changed = self._can_undo() and lf.undo.undo()
        except Exception:
            changed = False
        if changed:
            self._scene_generation = -1
            self._history_generation = -1
            self._refresh()

    def _on_redo_history(self, _handle, _event, _args):
        try:
            changed = self._can_redo() and lf.undo.redo()
        except Exception:
            changed = False
        if changed:
            self._scene_generation = -1
            self._history_generation = -1
            self._refresh()

    def _execute_delete_pipeline(self) -> str | None:
        try:
            result = lf.pipeline.edit.delete_().execute()
        except Exception as exc:
            return str(exc).strip() or _tr(
                "histogram.delete_failed.unexpected",
                "The delete pipeline raised an unexpected error.",
            )

        result_get = getattr(result, "get", None)
        if result_get is None:
            return _tr("histogram.delete_failed.invalid_result", "The delete pipeline returned an invalid result.")

        if bool(result_get("ok", False)):
            return None

        error = str(result_get("error", "") or "").strip()
        return error or _tr(
            "histogram.delete_failed.reported_failure",
            "The delete pipeline reported a failure.",
        )

    def _on_delete_marked(self, _handle, _event, _args):
        if not self._has_marked_range() or self._marked_count <= 0:
            return

        lo, hi = self._marked_bounds()
        if lo is None or hi is None or self._hist_edges is None:
            return

        self._apply_scene_selection(lo, hi)
        error_message = self._execute_delete_pipeline()
        if error_message is not None:
            self._status_hint = _trf(
                "histogram.delete_failed.status",
                "Delete failed: {message}",
                message=error_message,
            )
            if self._handle:
                self._handle.dirty_all()
            try:
                lf.ui.message_dialog(
                    _tr("histogram.delete_failed.title", "Delete Failed"),
                    error_message,
                    style="error",
                )
            except Exception:
                pass
            return

        self._clear_marked_range(clear_scene=False)
        self._scene_generation = -1
        self._refresh()

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _format_value(value: float) -> str:
        abs_value = abs(value)
        if abs_value == 0.0:
            return "0"
        if abs_value >= 10000.0 or abs_value < 1e-4:
            return f"{value:.2e}"
        if abs_value >= 100.0:
            return f"{value:.1f}"
        if abs_value >= 10.0:
            return f"{value:.2f}"
        if abs_value >= 1.0:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _format_range_text(self, range_min: float, range_max: float) -> str:
        return _trf(
            "histogram.range_value",
            "{min} to {max}",
            min=self._format_value(range_min),
            max=self._format_value(range_max),
        )

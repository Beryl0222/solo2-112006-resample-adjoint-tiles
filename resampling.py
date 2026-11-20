"""分片影像前向重采样与伴随反传引擎。

实现 ``交接协议.md`` 描述的合成交接格式：

* 坐标顺序 ``[y, x]``，像素中心位于整数坐标；``affine`` 把输出中心映射到源中心。
* 两轴独立的三角窗（半径来自 ``radius``），图外格点钳到最近合法格点，
  同一像素身份上的权重合并；零权重格点不计覆盖。
* ``null`` 为已作废观测（剔除后可重归一化），坐标无收件为缺报（仅可预览），
  实测零亮度是合法观测；同版本异值阻塞受影响的新格点。
* 前向返回不可变票据，票据只引用当次实际读到的源版本、有效像素与归一化权重；
  之后到达的源数据不会重新解释旧票据。
* 梯度按票据做伴随散射，归属到源像素行对应的拥有者；同一梯度批次重复送达
  不会重复累加。
* 滤波与伴随的临时张量全部经过 ``workspace`` 记账，按格点逐点处理，
  不物化整图；输入台账与各拥有者的最终分块结果不计入 workspace。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# 像素状态
VALID = "valid"          # 有合法观测值（含实测零亮度）
VOID = "void"            # 选中版本为已作废 null
MISSING = "missing"      # 可见修订上限内没有任何收件
CONFLICT = "conflict"    # 同版本异值

# 格点结局
TRAINABLE = "trainable"                 # 完整可训练
PREVIEW_MISSING = "preview_missing"     # 有缺报，仅预览（仍有部分有效值）
MISSING_NO_VALUE = "missing_no_value"   # 有权重的格点全缺，无值
NO_OBSERVATION = "no_observation"       # 作废剔除后无有效权重
BLOCKED_CONFLICT = "blocked_conflict"   # 同版本异值，阻塞

_REASON_TEXT = {
    TRAINABLE: "可训练",
    PREVIEW_MISSING: "存在缺报像素，仅可预览",
    MISSING_NO_VALUE: "有权重的格点全部缺报，返回无值",
    NO_OBSERVATION: "作废剔除后没有有效权重，标记无观测",
    BLOCKED_CONFLICT: "同版本异值冲突，格点阻塞",
}


class WorkspaceLimitExceeded(RuntimeError):
    """临时张量元素数超出协议约定的 workspace 上限。"""


class _Workspace:
    """滤波与伴随临时张量的元素记账器。

    只统计临时 ``numpy.ndarray`` 的元素数；输入台账、不可变票据以及落到各
    拥有者的最终分块结果都不是临时张量，不在此记账。
    """

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.live = 0
        self.peak = 0

    def allocate(self, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        size = 1
        for dim in shape:
            size *= int(dim)
        if self.live + size > self.limit:
            raise WorkspaceLimitExceeded(
                f"workspace 超限：需要 {size} 个元素，当前存活 {self.live}，"
                f"上限 {self.limit}"
            )
        self.live += size
        self.peak = max(self.peak, self.live)
        return np.empty(shape, dtype=dtype)

    def release(self, array: np.ndarray) -> None:
        self.live -= int(array.size)
        if self.live < 0:  # pragma: no cover - 防御性检查
            raise RuntimeError("workspace 记账失衡")


class HandoffEngine:
    """接收 tile/request/gradient 事件的重采样与伴随引擎。

    所有源图状态以稀疏字典保存（键为全局坐标 ``(y, x)``），任何阶段都不会
    按 ``shape`` 物化整图；临时数值张量按输出格点逐点申请，峰值元素数受
    ``workspace`` 约束。
    """

    def __init__(self, shape, owners, affine, radius, workspace):
        height, width = int(shape[0]), int(shape[1])
        owners = [str(owner) for owner in owners]
        if len(owners) != height:
            raise ValueError("owners 行数必须与源图高度一致")
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (2, 3):
            raise ValueError("affine 必须是 2×3 仿射矩阵")
        radius = (float(radius[0]), float(radius[1]))
        if radius[0] <= 0 or radius[1] <= 0:
            raise ValueError("滤波半径必须为正")

        self.shape = (height, width)
        self.owners = owners
        self.affine = affine
        self.radius = radius
        self.workspace = _Workspace(int(workspace))

        # (y, x) -> {修订号: [(tile 标识, 值或 None), ...]}
        self._ledger: dict[tuple[int, int], dict[int, list[tuple[str, float | None]]]] = {}
        self._tickets: dict[str, dict[str, Any]] = {}
        self._batches: dict[str, dict[str, Any]] = {}
        # 拥有者 -> {(y, x): 累积伴随梯度}，属最终分块结果，不计 workspace
        self._owner_grad: dict[str, dict[tuple[int, int], float]] = {
            owner: {} for owner in dict.fromkeys(owners)
        }

    # ------------------------------------------------------------------ tile

    def ingest_tile(self, tile_id, revision, observations):
        """接收一个 tile 的观测转发。

        同坐标同修订的一致转发记为重复（``null`` 之间也算一致）；出现两个及
        以上不同非空值时登记冲突，冲突在后续 request 评估时阻塞新格点。
        """
        tile_id = str(tile_id)
        revision = int(revision)
        height, width = self.shape
        summary = {
            "tile": tile_id,
            "revision": revision,
            "new": 0,
            "duplicate": 0,
            "void": 0,
            "conflicts": [],
        }
        for record in observations:
            if len(record) != 3:
                raise ValueError("tile 观测必须是 [y, x, value] 三元组")
            y, x, value = record
            if not (0 <= int(y) < height and 0 <= int(x) < width):
                raise ValueError(f"观测坐标 ({y}, {x}) 越界")
            coord = (int(y), int(x))
            value = None if value is None else float(value)
            pixel = self._ledger.setdefault(coord, {})
            entries = pixel.setdefault(revision, [])
            entries.append((tile_id, value))

            distinct = sorted({v for _, v in entries if v is not None})
            if value is None:
                summary["void"] += 1
            if len(distinct) >= 2:
                summary["conflicts"].append(
                    {"coord": list(coord), "revision": revision, "values": distinct}
                )
            elif any(v == value for tile, v in entries[:-1]) or (
                value is None and any(v is None for _, v in entries[:-1])
            ):
                summary["duplicate"] += 1
            else:
                summary["new"] += 1
        return summary

    # --------------------------------------------------------------- request

    def request(self, request_id, revision_cap, centers):
        """对一组输出中心做前向重采样，签发不可变票据并返回预览结果。"""
        request_id = str(request_id)
        if request_id in self._tickets:
            raise ValueError(f"票据标识 {request_id!r} 已存在")
        revision_cap = int(revision_cap)
        centers = [(int(y), int(x)) for y, x in centers]

        points = [
            self._evaluate_point(oy, ox, revision_cap)
            for oy, ox in centers
        ]
        ticket = {
            "id": request_id,
            "revision_cap": revision_cap,
            "centers": centers,
            "points": points,
        }
        self._tickets[request_id] = ticket
        return self._forward_view(ticket)

    def _evaluate_point(self, oy, ox, revision_cap):
        """评估单个输出格点：几何覆盖、像素状态、归一化与结局。"""
        sy = float(
            self.affine[0, 0] * oy + self.affine[0, 1] * ox + self.affine[0, 2]
        )
        sx = float(
            self.affine[1, 0] * oy + self.affine[1, 1] * ox + self.affine[1, 2]
        )
        radius_y, radius_x = self.radius
        height, width = self.shape

        # 几何窗口内的整数格点（严格 d < radius，零权重边界不计覆盖）。
        tap_y, weight_y, tracked_y = self._axis_taps(sy, radius_y)
        tap_x, weight_x, tracked_x = self._axis_taps(sx, radius_x)
        try:
            # 先按全局坐标认定唯一像素身份：图外格点钳位后合并权重。
            merged: dict[tuple[int, int], float] = {}
            order: list[tuple[int, int]] = []
            for iy in range(tap_y.size):
                source_y = int(tap_y[iy])
                clamped_y = min(max(source_y, 0), height - 1)
                wy = float(weight_y[iy])
                for ix in range(tap_x.size):
                    source_x = int(tap_x[ix])
                    clamped_x = min(max(source_x, 0), width - 1)
                    identity = (clamped_y, clamped_x)
                    if identity not in merged:
                        merged[identity] = 0.0
                        order.append(identity)
                    merged[identity] += wy * float(weight_x[ix])
        finally:
            # 标量降级路径的数组不在 workspace 记账内，不能释放。
            if tracked_x:
                self.workspace.release(weight_x)
                self.workspace.release(tap_x)
            if tracked_y:
                self.workspace.release(weight_y)
                self.workspace.release(tap_y)

        records = []
        for coord in order:
            state, revision, value = self._pixel_state(coord, revision_cap)
            records.append(
                {
                    "coord": list(coord),
                    "weight": float(merged[coord]),
                    "normalized_weight": None,
                    "state": state,
                    "revision": revision,
                    "value": value,
                }
            )

        conflicts = [
            {
                "coord": rec["coord"],
                "revision": rec["revision"],
                "values": self._conflict_values(tuple(rec["coord"]), rec["revision"]),
            }
            for rec in records
            if rec["state"] == CONFLICT
        ]
        valid_records = [rec for rec in records if rec["state"] == VALID]
        missing_pixels = [rec["coord"] for rec in records if rec["state"] == MISSING]
        void_pixels = [rec["coord"] for rec in records if rec["state"] == VOID]

        if conflicts:
            outcome = BLOCKED_CONFLICT
            value = None
        elif not valid_records:
            value = None
            outcome = MISSING_NO_VALUE if missing_pixels else NO_OBSERVATION
        else:
            value, total_weight = self._normalize(records, valid_records)
            for rec in valid_records:
                rec["normalized_weight"] = float(
                    np.float64(rec["weight"]) / total_weight
                )
            outcome = PREVIEW_MISSING if missing_pixels else TRAINABLE

        return {
            "center": [oy, ox],
            "source_center": [sy, sx],
            "outcome": outcome,
            "trainable": outcome == TRAINABLE,
            "reason": _REASON_TEXT[outcome],
            "value": value,
            "pixels": records,
            "valid_pixels": [rec["coord"] for rec in valid_records],
            "void_pixels": void_pixels,
            "missing_pixels": missing_pixels,
            "conflicts": conflicts,
        }

    def _axis_taps(self, center, radius):
        """返回某轴窗口内的整数格点坐标与三角窗权重。

        常规路径返回两张受记账张量；窗口大到放不下时退化为 Python 列表（非
        张量，不计 workspace），保证任何 workspace 上限下都能逐格点推进。
        """
        first = math.floor(center - radius) + 1
        last = math.ceil(center + radius) - 1
        coords = list(range(first, last + 1))
        count = len(coords)
        if 2 * count <= self.workspace.limit:
            taps = self.workspace.allocate((count,), np.int64)
            weights = self.workspace.allocate((count,), np.float64)
            try:
                for index, integer in enumerate(coords):
                    taps[index] = integer
                    weights[index] = self._triangle(center, integer, radius)
            except BaseException:  # pragma: no cover - 分配失败时立即归还
                self.workspace.release(weights)
                self.workspace.release(taps)
                raise
            return taps, weights, True

        # 标量路径：不产生临时张量。
        return (
            np.asarray(coords, dtype=np.int64),
            np.asarray(
                [self._triangle(center, integer, radius) for integer in coords],
                dtype=np.float64,
            ),
            False,
        )

    def _triangle(self, center, integer, radius):
        distance = abs(float(integer) - center)
        if distance >= radius:  # 零权重边界不计覆盖
            return np.float64(0.0)
        return np.float64(1.0) - np.float64(distance) / np.float64(radius)

    def _pixel_state(self, coord, revision_cap):
        """按可见修订上限评估单个源像素：取不超过上限的最高版本。"""
        versions = self._ledger.get(coord)
        if not versions:
            return MISSING, None, None
        visible = [revision for revision in versions if revision <= revision_cap]
        if not visible:
            return MISSING, None, None
        selected = max(visible)
        entries = versions[selected]
        distinct = sorted({value for _, value in entries if value is not None})
        if len(distinct) >= 2:
            return CONFLICT, selected, None
        if not distinct:
            return VOID, selected, None
        return VALID, selected, distinct[0]

    def _conflict_values(self, coord, revision):
        return sorted({
            value
            for _, value in self._ledger[coord][revision]
            if value is not None
        })

    def _normalize(self, records, valid_records):
        """作废/缺报剔除后按剩余有效权重重归一化，全程 FP64。

        常规路径使用两张长度为有效像素数的临时张量；若单格点覆盖过大，则
        退化为标量路径，保证任意情况下临时元素数都不突破 workspace。
        """
        n = len(records)
        weights = [np.float64(rec["weight"]) for rec in records]
        values = [
            np.float64(rec["value"]) if rec["state"] == VALID else np.float64("nan")
            for rec in records
        ]
        valid_index = [index for index, rec in enumerate(records) if rec["state"] == VALID]

        if 2 * n <= self.workspace.limit:
            weight_tensor = self.workspace.allocate((n,), np.float64)
            value_tensor = self.workspace.allocate((n,), np.float64)
            try:
                for index in range(n):
                    weight_tensor[index] = weights[index]
                    value_tensor[index] = values[index]
                total = np.float64(0.0)
                weighted = np.float64(0.0)
                for index in valid_index:
                    total += weight_tensor[index]
                    weighted += weight_tensor[index] * value_tensor[index]
            finally:
                self.workspace.release(value_tensor)
                self.workspace.release(weight_tensor)
        else:
            # 标量路径：只申请 1 元素的累加张量，仍然完全受 workspace 约束。
            total_tensor = self.workspace.allocate((1,), np.float64)
            weighted_tensor = self.workspace.allocate((1,), np.float64)
            try:
                total_tensor[0] = np.float64(0.0)
                weighted_tensor[0] = np.float64(0.0)
                for index in valid_index:
                    total_tensor[0] += weights[index]
                    weighted_tensor[0] += weights[index] * values[index]
                total = total_tensor[0]
                weighted = weighted_tensor[0]
            finally:
                self.workspace.release(weighted_tensor)
                self.workspace.release(total_tensor)

        if float(total) <= 0.0:
            return None, total
        return float(weighted / total), total

    def _forward_view(self, ticket):
        """构造面向调用者的前向返回值（纯 Python 数据，可 JSON 序列化）。"""
        points = []
        blocked = []
        for index, point in enumerate(ticket["points"]):
            view = {
                "index": index,
                "center": point["center"],
                "source_center": point["source_center"],
                "outcome": point["outcome"],
                "trainable": point["trainable"],
                "reason": point["reason"],
                "value": point["value"],
                "covered_pixels": [rec["coord"] for rec in point["pixels"]],
                "valid_pixels": point["valid_pixels"],
                "void_pixels": point["void_pixels"],
                "missing_pixels": point["missing_pixels"],
                "conflicts": point["conflicts"],
            }
            points.append(view)
            if not point["trainable"]:
                blocked.append(
                    {
                        "index": index,
                        "center": point["center"],
                        "reason": point["outcome"],
                        "detail": point["reason"],
                        "conflicts": point["conflicts"],
                        "missing_pixels": point["missing_pixels"],
                    }
                )
        return {
            "ticket": ticket["id"],
            "revision_cap": ticket["revision_cap"],
            "points": points,
            "trainable_indices": [
                index for index, point in enumerate(ticket["points"])
                if point["trainable"]
            ],
            "blocked": blocked,
            "peak_workspace": self.workspace.peak,
        }

    # -------------------------------------------------------------- gradient

    def apply_gradient(self, batch_id, ticket_id, gradients):
        """把一个梯度批次按票据伴随散射回源像素并归属拥有者。

        同一 ``batch_id`` 重复送达时直接返回首次结果的副本且不再累加；若重复
        载荷与首次不一致则报错。不可训练格点逐条返回拒绝信息，合法格点正常
        回传一次。
        """
        batch_id = str(batch_id)
        ticket_id = str(ticket_id)
        payload = [float(gradient) for gradient in gradients]

        if batch_id in self._batches:
            stored = self._batches[batch_id]
            if stored["ticket"] != ticket_id or stored["payload"] != payload:
                raise ValueError(f"梯度批次 {batch_id!r} 的重复送达与首次载荷不一致")
            duplicate_view = dict(stored["response"])
            duplicate_view["duplicate"] = True
            duplicate_view["applied"] = False
            return duplicate_view

        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise ValueError(f"未知票据 {ticket_id!r}")
        if len(payload) != len(ticket["points"]):
            raise ValueError("梯度长度必须与请求格点数一致")

        accepted = []
        rejected = []
        # 本次调用的归属增量：拥有者 -> 坐标 -> 增量
        delta_map: dict[str, dict[tuple[int, int], float]] = {}

        for index, (point, gradient) in enumerate(
            zip(ticket["points"], payload, strict=True)
        ):
            if not point["trainable"]:
                rejected.append(
                    {
                        "index": index,
                        "center": point["center"],
                        "reason": point["outcome"],
                        "detail": point["reason"],
                    }
                )
                continue

            valid_records = [
                rec for rec in point["pixels"] if rec["state"] == VALID
            ]
            count = len(valid_records)
            if count <= self.workspace.limit:
                contributions = self.workspace.allocate((count,), np.float64)
                try:
                    for k, rec in enumerate(valid_records):
                        contributions[k] = np.float64(
                            rec["normalized_weight"]
                        ) * np.float64(gradient)
                    for k, rec in enumerate(valid_records):
                        self._scatter(rec, float(contributions[k]), delta_map)
                finally:
                    self.workspace.release(contributions)
            else:  # 标量路径，保证 workspace 硬上限
                for rec in valid_records:
                    delta = float(
                        np.float64(rec["normalized_weight"]) * np.float64(gradient)
                    )
                    self._scatter(rec, delta, delta_map)

            accepted.append(
                {"index": index, "center": point["center"], "gradient": gradient}
            )

        attribution = self._attribution_view(delta_map)
        response = {
            "batch": batch_id,
            "ticket": ticket_id,
            "duplicate": False,
            "applied": True,
            "accepted": accepted,
            "rejected": rejected,
            "attribution": attribution,
            "peak_workspace": self.workspace.peak,
        }
        self._batches[batch_id] = {
            "ticket": ticket_id,
            "payload": payload,
            "response": response,
        }
        return response

    def _scatter(self, record, delta, delta_map):
        y, x = record["coord"]
        owner = self.owners[y]
        coord = (y, x)
        owner_bucket = self._owner_grad[owner]
        owner_bucket[coord] = owner_bucket.get(coord, 0.0) + np.float64(delta)
        call_bucket = delta_map.setdefault(owner, {})
        call_bucket[coord] = call_bucket.get(coord, 0.0) + np.float64(delta)

    def _attribution_view(self, delta_map):
        """把本次增量与累积值整理成按拥有者划分的稀疏分块。"""
        view = {}
        for owner in self._owner_grad:
            deltas = delta_map.get(owner, {})
            if not deltas:
                continue
            coords = sorted(deltas)
            view[owner] = {
                "pixels": [list(coord) for coord in coords],
                "delta": [float(deltas[coord]) for coord in coords],
                "cumulative": [
                    float(self._owner_grad[owner][coord]) for coord in coords
                ],
            }
        return view

    def owner_chunks(self):
        """导出各拥有者的最终稀疏梯度分块（不计 workspace）。"""
        return {
            owner: [
                [coord[0], coord[1], float(gradient)]
                for coord, gradient in sorted(bucket.items())
            ]
            for owner, bucket in self._owner_grad.items()
            if bucket
        }

    # ------------------------------------------------------------------ replay

    @classmethod
    def from_capture(cls, capture):
        return cls(
            capture["shape"],
            capture["owners"],
            capture["affine"],
            capture["radius"],
            capture["workspace"],
        )


def replay(capture):
    """按采集文件中的原始事件顺序回放，返回引擎与每个事件的返回值。"""
    engine = HandoffEngine.from_capture(capture)
    results = []
    for event in capture["events"]:
        if not isinstance(event, list) or not event:
            raise ValueError("采集资料含有无效事件")
        kind = event[0]
        if kind == "tile":
            _, tile_id, revision, observations = event
            results.append(
                ("tile", engine.ingest_tile(tile_id, revision, observations))
            )
        elif kind == "request":
            _, request_id, revision_cap, centers = event
            results.append(
                ("request", engine.request(request_id, revision_cap, centers))
            )
        elif kind == "gradient":
            _, batch_id, ticket_id, gradients = event
            results.append(
                (
                    "gradient",
                    engine.apply_gradient(batch_id, ticket_id, gradients),
                )
            )
        else:
            raise ValueError(f"未知事件类型 {kind!r}")
    return engine, results


def _format_summary(kind, result):
    if kind == "tile":
        conflict_text = (
            "；冲突 " + ", ".join(
                f"{item['coord']} rev{item['revision']}={item['values']}"
                for item in result["conflicts"]
            )
            if result["conflicts"]
            else ""
        )
        return (
            f"tile {result['tile']} rev={result['revision']}："
            f"新增 {result['new']} 重复 {result['duplicate']} "
            f"作废 {result['void']}{conflict_text}"
        )
    if kind == "request":
        lines = [
            f"request {result['ticket']} (cap={result['revision_cap']})："
            f"可训练格点 {result['trainable_indices']}，阻塞 {len(result['blocked'])}"
        ]
        for point in result["points"]:
            value = "无值" if point["value"] is None else f"{point['value']:.12g}"
            lines.append(
                f"  格点{point['index']} 中心{tuple(point['center'])} "
                f"{point['outcome']} 值={value} "
                f"覆盖{len(point['covered_pixels'])} "
                f"有效{len(point['valid_pixels'])} "
                f"作废{len(point['void_pixels'])} "
                f"缺报{len(point['missing_pixels'])} "
                f"冲突{len(point['conflicts'])}"
            )
        return "\n".join(lines)
    lines = [
        f"gradient {result['batch']} -> {result['ticket']}："
        f"applied={result['applied']} duplicate={result['duplicate']} "
        f"接受 {len(result['accepted'])} 拒绝 {len(result['rejected'])}"
    ]
    for rejection in result["rejected"]:
        lines.append(
            f"  拒绝 格点{rejection['index']} 中心{tuple(rejection['center'])} "
            f"{rejection['reason']}"
        )
    for owner, chunk in sorted(result["attribution"].items()):
        lines.append(
            f"  归属 {owner}: {len(chunk['pixels'])} 个源像素 "
            f"delta={[round(v, 9) for v in chunk['delta']]}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="回放交接采集并打印前向/反传结果")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args(argv)
    with args.capture.open(encoding="utf-8") as stream:
        capture = json.load(stream)
    _, results = replay(capture)
    for index, (kind, result) in enumerate(results, start=1):
        print(f"{index:02} {_format_summary(kind, result)}")


if __name__ == "__main__":
    main()

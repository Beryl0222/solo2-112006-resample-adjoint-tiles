"""resampling 引擎的协议语义、数值口径与 workspace 测试。"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resampling import (  # noqa: E402
    BLOCKED_CONFLICT,
    MISSING_NO_VALUE,
    NO_OBSERVATION,
    PREVIEW_MISSING,
    TRAINABLE,
    HandoffEngine,
    WorkspaceLimitExceeded,
    replay,
)

CAPTURE = ROOT / "samples" / "capture.json"
TOL = 1e-6


# ---------------------------------------------------------------------- 参考实现


def triangle(center, integer, radius):
    distance = abs(float(integer) - center)
    if distance >= radius:
        return 0.0
    return 1.0 - distance / radius


def reference_forward(shape, affine, radius, source, center, select):
    """独立的稠密参考前向：钳位合并、三角窗、按 select 的像素状态重归一化。"""
    height, width = shape
    oy, ox = center
    sy = affine[0, 0] * oy + affine[0, 1] * ox + affine[0, 2]
    sx = affine[1, 0] * oy + affine[1, 1] * ox + affine[1, 2]
    ry, rx = radius

    merged = {}
    iy0 = int(np.floor(sy - ry)) + 1
    iy1 = int(np.ceil(sy + ry)) - 1
    ix0 = int(np.floor(sx - rx)) + 1
    ix1 = int(np.ceil(sx + rx)) - 1
    for iy in range(iy0, iy1 + 1):
        wy = triangle(sy, iy, ry)
        if wy <= 0.0:
            continue
        cy = min(max(iy, 0), height - 1)
        for ix in range(ix0, ix1 + 1):
            wx = triangle(sx, ix, rx)
            if wx <= 0.0:
                continue
            cx = min(max(ix, 0), width - 1)
            key = (cy, cx)
            merged[key] = merged.get(key, 0.0) + wy * wx

    weights, values = [], []
    for (y, x), weight in merged.items():
        state = select(y, x)
        if state == "valid":
            weights.append(np.float64(weight))
            values.append(np.float64(source[y, x]))
    if not weights:
        return None, merged
    total = np.float64(0.0)
    weighted = np.float64(0.0)
    for weight, value in zip(weights, values, strict=True):
        total += weight
        weighted += weight * value
    return float(weighted / total), merged


def build_full_engine(shape, owners, affine, radius, workspace, rng):
    """构造一个数据齐全的引擎与稠密源图，值覆盖零亮度。"""
    height, width = shape
    source = rng.random((height, width)) * 9.0
    source[0, 0] = 0.0
    engine = HandoffEngine(shape, owners, affine, radius, workspace)
    observations = [
        [y, x, float(source[y, x])]
        for y in range(height)
        for x in range(width)
    ]
    engine.ingest_tile("full", 1, observations)
    return engine, source


# ---------------------------------------------------------------------- 样例语义


class CaptureSemanticsTests(unittest.TestCase):
    def setUp(self):
        with CAPTURE.open(encoding="utf-8") as stream:
            self.capture = json.load(stream)
        self.engine, self.results = replay(self.capture)

    def test_event_result_kinds(self):
        self.assertEqual([kind for kind, _ in self.results],
                         ["tile", "request", "tile", "request", "tile",
                          "request", "gradient", "gradient", "gradient"])

    def test_p0_outcomes_and_ranges(self):
        view = self.results[1][1]
        self.assertEqual(view["trainable_indices"], [0])
        outcomes = [point["outcome"] for point in view["points"]]
        self.assertEqual(outcomes, [TRAINABLE, PREVIEW_MISSING])
        # 缺报格点仍有预览值；作废像素被剔除。
        self.assertIsNotNone(view["points"][1]["value"])
        point0 = view["points"][0]
        self.assertEqual(len(point0["void_pixels"]), 1)
        self.assertEqual(point0["missing_pixels"], [])

    def test_conflicting_tile_is_recorded(self):
        tile_c = self.results[4][1]
        self.assertEqual(tile_c["new"], 0)
        self.assertEqual(tile_c["duplicate"], 0)
        self.assertEqual(tile_c["conflicts"],
                         [{"coord": [1, 2], "revision": 1, "values": [6.0, 18.0]}])

    def test_p1_then_p2_immutability(self):
        p1 = self.results[3][1]
        self.assertEqual(p1["trainable_indices"], [0])
        p1_value = p1["points"][0]["value"]
        p2 = self.results[5][1]
        self.assertEqual(p2["trainable_indices"], [])
        self.assertEqual(p2["points"][0]["outcome"], BLOCKED_CONFLICT)
        # 旧票据不因后到数据被重新解释。
        self.assertEqual(self.engine._tickets["p1"]["points"][0]["value"], p1_value)

    def test_gradient_rejection_acceptance_and_duplicate(self):
        g0 = self.results[6][1]
        self.assertTrue(g0["applied"])
        self.assertEqual([item["index"] for item in g0["accepted"]], [0])
        self.assertEqual([item["index"] for item in g0["rejected"]], [1])
        self.assertEqual(g0["rejected"][0]["reason"], PREVIEW_MISSING)

        g1_first = self.results[7][1]
        self.assertTrue(g1_first["applied"])
        chunks_after_first = {
            owner: [row[:] for row in chunk]
            for owner, chunk in self.engine.owner_chunks().items()
        }
        g1_again = self.results[8][1]
        self.assertTrue(g1_again["duplicate"])
        self.assertFalse(g1_again["applied"])
        # 重复批次不得再给重叠源像素加一次。
        self.assertEqual(self.engine.owner_chunks(), chunks_after_first)

    def test_attribution_uses_row_owners(self):
        chunks = self.engine.owner_chunks()
        for owner, rows in chunks.items():
            for y, x, _ in rows:
                self.assertEqual(self.capture["owners"][y], owner)
        # 冲突格点 p2 未训练，反传后仍无 (1,2)=18 的痕迹；(1,2) 仅来自 p0/p1。
        self.assertIn([1, 2, 0.6386657203639642], chunks["north"])

    def test_workspace_peak_within_protocol_limit(self):
        self.assertLessEqual(self.engine.workspace.peak, self.capture["workspace"])
        self.assertEqual(self.engine.workspace.live, 0)

    def test_ticket_facts_are_snapshotted(self):
        p1_ticket = self.engine._tickets["p1"]
        record = next(
            rec for rec in p1_ticket["points"][0]["pixels"]
            if rec["coord"] == [1, 2]
        )
        self.assertEqual(record["state"], "valid")
        self.assertEqual(record["value"], 6.0)
        self.assertIsNotNone(record["normalized_weight"])


# ---------------------------------------------------------------------- 数值与伴随


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.shape = (6, 8)
        self.owners = ["r0", "r0", "r1", "r1", "r2", "r2"]
        self.affine = np.array(
            [[1.15, 0.35, 0.35], [-0.25, 1.55, 0.4]], dtype=np.float64
        )
        self.radius = (1.5, 1.75)
        self.workspace = 40
        self.engine, self.source = build_full_engine(
            self.shape, self.owners, self.affine, self.radius,
            self.workspace, self.rng,
        )

    def test_forward_matches_independent_reference(self):
        centers = [(0, 0), (1, 1), (3, 4), (5, 7)]
        view = self.engine.request("ref", 1, centers)
        for index, center in enumerate(centers):
            expected, _ = reference_forward(
                self.shape, self.affine, self.radius, self.source, center,
                lambda y, x: "valid",
            )
            np.testing.assert_allclose(
                view["points"][index]["value"], expected, rtol=TOL, atol=TOL
            )
            self.assertEqual(view["points"][index]["outcome"], TRAINABLE)

    def test_adjoint_inner_product_identity(self):
        centers = [(0, 0), (1, 2), (2, 3), (4, 5)]
        self.engine.request("adj", 1, centers)
        upstream = self.rng.standard_normal(len(centers))
        response = self.engine.apply_gradient("bg", "adj", upstream.tolist())

        # 参考雅可比：重算每个格点的归一化权重矩阵。
        lhs = np.float64(0.0)
        for index, center in enumerate(centers):
            _, merged = reference_forward(
                self.shape, self.affine, self.radius, self.source, center,
                lambda y, x: "valid",
            )
            total = sum(merged.values())
            for (y, x), weight in merged.items():
                lhs += np.float64(upstream[index]) * np.float64(weight / total) \
                    * np.float64(self.source[y, x])

        rhs = np.float64(0.0)
        for rows in self.engine.owner_chunks().values():
            for y, x, gradient in rows:
                rhs += np.float64(gradient) * np.float64(self.source[y, x])
        np.testing.assert_allclose(rhs, lhs, rtol=1e-9, atol=1e-9)
        self.assertEqual(len(response["rejected"]), 0)

    def test_adjoint_against_explicit_jacobian(self):
        centers = [(0, 1), (2, 2), (3, 6)]
        self.engine.request("jac", 1, centers)
        upstream = np.array([1.5, -2.0, 0.75])
        self.engine.apply_gradient("bj", "jac", upstream.tolist())

        height, width = self.shape
        expected = np.zeros((height, width), dtype=np.float64)
        for index, center in enumerate(centers):
            _, merged = reference_forward(
                self.shape, self.affine, self.radius, self.source, center,
                lambda y, x: "valid",
            )
            total = sum(merged.values())
            for (y, x), weight in merged.items():
                expected[y, x] += upstream[index] * weight / total

        actual = np.zeros((height, width), dtype=np.float64)
        for rows in self.engine.owner_chunks().values():
            for y, x, gradient in rows:
                actual[y, x] = gradient
        np.testing.assert_allclose(actual, expected, rtol=TOL, atol=TOL)

    def test_tiling_invariance_forward_and_adjoint(self):
        centers = [(y, x) for y in range(6) for x in range(8)]
        full_view = self.engine.request("full", 1, centers)
        full_up = self.rng.standard_normal(len(centers))
        full_resp = self.engine.apply_gradient(
            "bf", "full", full_up.tolist()
        )

        # 改变切块边界：同一组格点切成三块，逐块前向与反传。
        engine2, _ = build_full_engine(
            self.shape, self.owners, self.affine, self.radius,
            self.workspace, np.random.default_rng(7),
        )
        cuts = [0, 13, 31, len(centers)]
        split_values = []
        for part in range(3):
            sub = centers[cuts[part]:cuts[part + 1]]
            sub_view = engine2.request(f"part{part}", 1, sub)
            split_values.extend(
                point["value"] for point in sub_view["points"]
            )
            engine2.apply_gradient(
                f"bp{part}", f"part{part}",
                full_up[cuts[part]:cuts[part + 1]].tolist(),
            )

        for full_point, split_value in zip(full_view["points"], split_values):
            np.testing.assert_allclose(
                full_point["value"], split_value, rtol=TOL, atol=TOL
            )
        for owner in full_resp["attribution"]:
            a = dict(
                zip(
                    [tuple(p) for p in full_resp["attribution"][owner]["pixels"]],
                    full_resp["attribution"][owner]["cumulative"],
                )
            )
            b = {
                (y, x): gradient
                for y, x, gradient in engine2.owner_chunks()[owner]
            }
            self.assertEqual(set(a), set(b))
            for coord in a:
                np.testing.assert_allclose(a[coord], b[coord], rtol=TOL, atol=TOL)


# ---------------------------------------------------------------------- 像素语义


class PixelSemanticsTests(unittest.TestCase):
    SHAPE = (4, 5)

    def _engine(self, workspace=40):
        owners = ["n", "n", "s", "s"]
        # 输出 (0,0) 的源中心落在源像素 (1,1)；半径组合 (1.5, 1.0) 使
        # y 轴覆盖 {0,1,2} 行、x 轴只覆盖中心列，窗内恰为十字五像素
        # (半径 1.0 时严格边界外零权重，由单独的边界测试覆盖)。
        affine = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        return HandoffEngine(self.SHAPE, owners, affine, (1.5, 1.0), workspace)

    def _neighborhood(self, values):
        """以 (1,1) 为中心的十字窗五个像素；values 为 coord -> value/None。"""
        coords = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
        return [[y, x, values[(y, x)]] for (y, x) in coords if (y, x) in values]

    def test_zero_brightness_is_a_valid_observation(self):
        engine = self._engine()
        values = {
            (1, 1): 0.0, (0, 1): 4.0, (2, 1): 2.0, (1, 0): 6.0, (1, 2): 8.0,
        }
        engine.ingest_tile("z", 1, self._neighborhood(values))
        view = engine.request("r", 1, [(0, 0)])
        point = view["points"][0]
        self.assertEqual(point["outcome"], TRAINABLE)
        # 半径 (1.5, 1.0)：x 轴邻域恰在零权重边界，只有中心与 y 轴邻域参与；
        # 权重为 1/3、1、1/3，零亮度的中心作为合法观测把均值拉低。
        expected = (4.0 / 3.0 + 2.0 / 3.0) / (1.0 + 2.0 / 3.0)
        np.testing.assert_allclose(point["value"], expected, rtol=TOL, atol=TOL)
        center = next(
            rec for rec in engine._tickets["r"]["points"][0]["pixels"]
            if rec["coord"] == [1, 1]
        )
        self.assertEqual(center["state"], "valid")
        self.assertEqual(center["value"], 0.0)

    def test_void_pixels_are_renormalized(self):
        engine = self._engine()
        values = {
            (1, 1): None, (0, 1): 8.0, (2, 1): 8.0, (1, 0): 8.0, (1, 2): 8.0,
        }
        engine.ingest_tile("v", 1, self._neighborhood(values))
        view = engine.request("r", 1, [(0, 0)])
        point = view["points"][0]
        self.assertEqual(point["outcome"], TRAINABLE)
        self.assertEqual(point["void_pixels"], [[1, 1]])
        # 四个等权邻域像素均为 8，重归一化后仍为 8。
        np.testing.assert_allclose(point["value"], 8.0, rtol=TOL, atol=TOL)

    def test_all_void_means_no_observation(self):
        engine = self._engine()
        values = {(1, 1): None, (0, 1): None, (2, 1): None,
                  (1, 0): None, (1, 2): None}
        engine.ingest_tile("v", 1, self._neighborhood(values))
        view = engine.request("r", 1, [(0, 0)])
        self.assertEqual(view["points"][0]["outcome"], NO_OBSERVATION)
        self.assertIsNone(view["points"][0]["value"])
        rejected = engine.apply_gradient("g", "r", [1.0])["rejected"]
        self.assertEqual(rejected[0]["reason"], NO_OBSERVATION)

    def test_center_only_window_all_void_is_no_observation(self):
        # 半径 1：严格边界外零权重，窗口只剩中心像素；中心作废即无观测。
        engine = HandoffEngine(
            self.SHAPE, ["n", "n", "s", "s"],
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            (1.0, 1.0), 40,
        )
        engine.ingest_tile("v", 1, [[1, 1, None]])
        view = engine.request("r", 1, [(0, 0)])
        self.assertEqual(view["points"][0]["covered_pixels"], [[1, 1]])
        self.assertEqual(view["points"][0]["outcome"], NO_OBSERVATION)

    def test_missing_returns_preview_and_full_missing_no_value(self):
        engine = self._engine()
        # 窗内为十字五像素；中心与 x 轴邻域有值，y 轴上下两个邻域缺报。
        values = {(1, 1): 2.0, (1, 0): 4.0, (1, 2): 6.0}
        engine.ingest_tile("m", 1, self._neighborhood(values))
        partial = engine.request("partial", 1, [(0, 0)])
        point = partial["points"][0]
        self.assertEqual(point["outcome"], PREVIEW_MISSING)
        self.assertIsNotNone(point["value"])
        self.assertEqual(sorted(point["missing_pixels"]), [[0, 1], [2, 1]])

        empty = self._engine()
        none_view = empty.request("none", 1, [(0, 0)])
        self.assertEqual(none_view["points"][0]["outcome"], MISSING_NO_VALUE)
        self.assertIsNone(none_view["points"][0]["value"])

    def test_revision_cap_visibility_and_upgrade(self):
        engine = self._engine()
        values = {(1, 1): 1.0, (0, 1): 3.0, (2, 1): 5.0,
                  (1, 0): 7.0, (1, 2): 9.0}
        engine.ingest_tile("old", 1, self._neighborhood(values))
        old_view = engine.request("oldreq", 1, [(0, 0)])
        self.assertEqual(old_view["points"][0]["outcome"], TRAINABLE)
        old_value = old_view["points"][0]["value"]
        # 修订号 2 到达后，旧 cap 票据仍引用版本 1 的事实。
        engine.ingest_tile("new", 2, [[1, 1, 100.0]])
        self.assertEqual(
            engine._tickets["oldreq"]["points"][0]["pixels"][0]["revision"], 1
        )
        new_view = engine.request("newreq", 2, [(0, 0)])
        self.assertNotAlmostEqual(new_view["points"][0]["value"], old_value)
        # cap=1 的新请求看不到修订 2，结果与旧票据一致。
        capped = engine.request("cap1", 1, [(0, 0)])
        np.testing.assert_allclose(
            capped["points"][0]["value"], old_value, rtol=TOL, atol=TOL
        )

    def test_same_value_forwarding_is_merged_not_conflict(self):
        engine = self._engine()
        values = {(1, 1): 6.0, (0, 1): 4.0, (2, 1): 2.0,
                  (1, 0): 8.0, (1, 2): 1.0}
        engine.ingest_tile("a", 1, self._neighborhood(values))
        second = engine.ingest_tile("b", 1, [[1, 1, 6.0]])
        self.assertEqual(second["duplicate"], 1)
        self.assertEqual(second["conflicts"], [])
        view = engine.request("r", 1, [(0, 0)])
        self.assertEqual(view["points"][0]["outcome"], TRAINABLE)

    def test_conflict_blocks_new_request_but_old_ticket_survives(self):
        engine = self._engine()
        values = {(1, 1): 6.0, (0, 1): 4.0, (2, 1): 2.0,
                  (1, 0): 8.0, (1, 2): 1.0}
        engine.ingest_tile("a", 1, self._neighborhood(values))
        before = engine.request("before", 1, [(0, 0)])
        self.assertEqual(before["points"][0]["outcome"], TRAINABLE)
        engine.ingest_tile("c", 1, [[1, 1, 18.0]])
        after = engine.request("after", 1, [(0, 0)])
        self.assertEqual(after["points"][0]["outcome"], BLOCKED_CONFLICT)
        # 早期一致输入上固定的反传仍能交接。
        response = engine.apply_gradient("g", "before", [2.5])
        self.assertEqual(len(response["accepted"]), 1)
        blocked = engine.apply_gradient("g2", "after", [2.5])
        self.assertEqual(len(blocked["accepted"]), 0)
        self.assertEqual(blocked["rejected"][0]["reason"], BLOCKED_CONFLICT)

    def test_clamp_merges_out_of_bounds_identities(self):
        # 半径 0.5：输出中心 (-1,-1) 的源中心为 (0,0)，两轴窗只覆盖 0 与 -1，
        # 图外的 -1 钳到 0，唯一像素身份是 (0,0)，钳位权重合并到该身份。
        engine = HandoffEngine(
            self.SHAPE, ["n", "n", "s", "s"],
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            (0.5, 0.5), 40,
        )
        engine.ingest_tile("edge", 1, [[0, 0, 3.0]])
        view = engine.request("edge", 1, [(-1, -1)])
        point = view["points"][0]
        self.assertEqual(point["covered_pixels"], [[0, 0]])
        self.assertEqual(point["outcome"], TRAINABLE)
        np.testing.assert_allclose(point["value"], 3.0, rtol=TOL, atol=TOL)


# ---------------------------------------------------------------------- 运行约束


class WorkspaceAndStreamingTests(unittest.TestCase):
    def test_tiny_workspace_scalar_fallback_matches(self):
        rng = np.random.default_rng(11)
        shape = (5, 6)
        owners = ["a", "a", "b", "b", "c"]
        affine = np.array([[1.2, 0.1, 0.3], [0.2, 1.3, 0.7]])
        engine, source = build_full_engine(
            shape, owners, affine, (1.5, 1.75), 2, rng
        )
        view = engine.request("tiny", 1, [(1, 1), (3, 4)])
        for index, center in enumerate([(1, 1), (3, 4)]):
            expected, _ = reference_forward(
                shape, affine, (1.5, 1.75), source, center,
                lambda y, x: "valid",
            )
            np.testing.assert_allclose(
                view["points"][index]["value"], expected, rtol=TOL, atol=TOL
            )
        self.assertLessEqual(engine.workspace.peak, 2)
        response = engine.apply_gradient("g", "tiny", [1.0, 2.0])
        self.assertEqual(engine.workspace.live, 0)
        self.assertEqual(len(response["accepted"]), 2)

    def test_zero_workspace_rejects_tensor_allocation(self):
        engine = HandoffEngine(
            (3, 3), ["a", "a", "a"],
            np.eye(2, 3), (1.0, 1.0), 0,
        )
        engine.ingest_tile("t", 1, [[0, 0, 1.0]])
        with self.assertRaises(WorkspaceLimitExceeded):
            engine.request("r", 1, [(0, 0)])

    def test_large_image_uses_sparse_state_only(self):
        rng = np.random.default_rng(3)
        height, width = 300, 250
        owners = [f"row{y % 5}" for y in range(height)]
        # 仿射平移使输出 (oy, ox) 的源中心恰好为 (oy, ox)（单位映射），
        # 请求中心直接取送达条带内的全局坐标，窗口完全落在条带内。
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        engine = HandoffEngine(
            (height, width), owners, affine, (1.0, 1.0), 16,
        )
        # 只稀疏送达一个条带的数据，请求中心全部落在条带附近。
        observations = [
            [y, x, float(rng.random())]
            for y in range(100, 104)
            for x in range(40, 60)
        ]
        engine.ingest_tile("strip", 1, observations)
        centers = [(y, x) for y in range(100, 104) for x in range(42, 56)]
        view = engine.request("stripreq", 1, centers)
        self.assertLessEqual(engine.workspace.peak, 16)
        trainable = [p["index"] for p in view["points"] if p["trainable"]]
        self.assertTrue(trainable)
        response = engine.apply_gradient(
            "stripg", "stripreq",
            [1.0 if i in trainable else 0.0 for i in range(len(centers))],
        )
        self.assertEqual(engine.workspace.live, 0)
        touched = sum(
            len(chunk) for chunk in response["attribution"].values()
        )
        # 只触及送达条带内的像素。
        self.assertLessEqual(touched, len(observations))

    def test_command_line_replay(self):
        result = subprocess.run(
            [sys.executable, "-m", "resampling", str(CAPTURE)],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        # 九个事件各有一个编号标题，多格点请求与梯度归属占多行明细。
        headers = [
            line for line in result.stdout.splitlines()
            if len(line) >= 2 and line[:2].isdigit()
        ]
        self.assertEqual(len(headers), 9)
        self.assertIn("blocked_conflict", result.stdout)
        self.assertIn("duplicate=True", result.stdout)


if __name__ == "__main__":
    unittest.main()

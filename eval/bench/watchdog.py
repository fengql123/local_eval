"""Bound candidate measurement using accepted tokens and active serving time."""

from __future__ import annotations

import asyncio
from collections import deque
from time import perf_counter

from .metrics import combined_gain, gains

WINDOW = 120.0


class PerformanceCutoff(RuntimeError):
    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail["reason"])


class Watchdog:
    """Observe streams without interpreting SSE events as generated token counts.

    N workloads include prefill/queue time. Balanced workloads use decode time
    and observed mean TTFT, with a separate no-first-token allowance. Tool-only
    waits have no active requests and consume no performance window.
    """

    def __init__(self, workload, *, reference=None, floor=None, window=WINDOW, clock=perf_counter):
        self.workload, self.reference, self.floor = workload, reference, floor
        self.window, self.clock = window, clock
        self.active = {}
        self.previous = clock()
        self.elapsed = self.tokens = self.ttft_sum = self.ttft_count = 0
        self.samples = deque([(0.0, 0)])
        self.failed = False

    def _tick(self):
        now = self.clock()
        decoding = any(state["first"] is not None for state in self.active.values())
        if self.active and (self.workload in ("S-N", "A-N") or decoding):
            self.elapsed += now - self.previous
        self.previous = now
        return now

    def start(self, result):
        self._tick()
        self.active[id(result)] = {"count": 0, "first": None, "start": self.clock()}

    def update(self, result):
        self._tick()
        state = self.active[id(result)]
        count = len(result["token_ids"]) if "token_ids" in result else result.get("output_tokens")
        if isinstance(count, int) and count >= state["count"]:
            self.tokens += count - state["count"]
            state["count"] = count
        first = result.get("t_first")
        if first is not None and state["first"] is None:
            state["first"] = first
            self.ttft_sum += first - result["t_send"]
            self.ttft_count += 1

    def finish(self, result):
        self.update(result)
        del self.active[id(result)]

    def check(self):
        now = self._tick()
        if self.failed or not self.active or (self.reference is None and self.floor is None):
            return None
        if self.workload not in ("S-N", "A-N"):
            allowance = (
                max(self.window, 2 * self.reference["ttft_mean"]) if self.reference else self.window
            )
            if any(
                s["first"] is None and now - s["start"] >= allowance for s in self.active.values()
            ):
                raise PerformanceCutoff(
                    {
                        "reason": "first-token performance allowance exceeded",
                        "first_token_timeout_seconds": allowance,
                    }
                )
        self.samples.append((self.elapsed, self.tokens))
        target = self.elapsed - self.window
        while len(self.samples) > 1 and self.samples[1][0] <= target:
            self.samples.popleft()
        elapsed, tokens = self.samples[0]
        span = self.elapsed - elapsed
        if span < self.window:
            return None
        measured = {"throughput": (self.tokens - tokens) / span}
        overall = {"throughput": self.tokens / self.elapsed}
        if self.workload not in ("S-N", "A-N"):
            measured["ttft_mean"] = self.ttft_sum / self.ttft_count if self.ttft_count else None
            overall["ttft_mean"] = measured["ttft_mean"]
        if self.reference:
            value, threshold = combined_gain(gains(self.workload, self.reference, measured)), 1.0
            overall_value = combined_gain(gains(self.workload, self.reference, overall))
            reason = "recent and overall scored gain stayed at or below reference parity"
        else:
            value, threshold = measured["throughput"], self.floor
            overall_value = overall["throughput"]
            reason = "recent and overall throughput stayed below the lowest SLO tier"
        detail = {
            "reason": reason,
            "window_seconds": span,
            "observed": value,
            "threshold": threshold,
            "overall_observed": overall_value,
            "overall_throughput": overall["throughput"],
            **measured,
        }
        # A draining N batch naturally has fewer active requests. A slower tail
        # must not reject an operation whose measured overall performance wins.
        under = (
            value <= threshold and overall_value <= threshold
            if self.reference
            else value < threshold and overall_value < threshold
        )
        if under:
            raise PerformanceCutoff(detail)
        return detail

    async def run(self, progress=None):
        reported = self.clock()
        while True:
            await asyncio.sleep(1)
            detail = self.check()
            if progress and self.clock() - reported >= 5:
                progress.event(
                    "serving_progress",
                    active_requests=len(self.active),
                    accepted_tokens=self.tokens,
                    active_seconds=self.elapsed,
                )
                reported = self.clock()
            if detail and progress:
                progress.event("performance_sample", **detail)

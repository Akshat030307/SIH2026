from smartscan.runner import run_episode
from smartscan.schedulers.baselines import LinearSweep
from smartscan.schedulers.heuristic import SmartHeuristic
from smartscan.sim.world import build_world


def test_smart_heuristic_beats_sweep_on_lockin():
    w = build_world("S6_lockin", seed=0)
    m_sweep, _, _ = run_episode(w, LinearSweep(), seed=0)
    m_smart, _, _ = run_episode(build_world("S6_lockin", seed=0), SmartHeuristic(), seed=0)
    assert m_sweep["pd"] < 0.05
    assert m_smart["pd"] > 0.5
    assert m_smart["predicted_dwells"] > 0

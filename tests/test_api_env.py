import numpy as np

from smartscan.api import SmartScanScheduler
from smartscan.rl.env import SmartScanEnv
from smartscan.rl.perception import F_CH, F_GLOBAL
from smartscan.sim.engine import RFEngine
from smartscan.sim.world import build_world


def test_api_drives_a_receiver_loop():
    world = build_world("S1_sparse", seed=3)
    eng = RFEngine(world)
    api = SmartScanScheduler({"f_min_ghz": 2, "f_max_ghz": 18, "ibw_mhz": 500}, policy="smart")
    for _ in range(400):
        cmd = api.next_command(eng.t)
        assert 2e9 <= cmd.center_freq_hz <= 18e9 and cmd.dwell_s > 0
        res = eng.step(cmd.channel, api.rx.dwells_s.index(cmd.dwell_s))
        cols = np.stack([res.pdws[k] for k in ("toa", "rf", "pw", "aoa", "amp")], axis=1)
        ids = api.ingest(cols, res.t_listen, res.t_end)  # plain (N, 5) array input
        assert len(ids) == len(res.pdws)
    assert len(api.tracks()) > 0


def test_env_observation_and_expert():
    env = SmartScanEnv(scenarios=["S6_lockin"], seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (env.K * F_CH + F_GLOBAL,)
    assert np.isfinite(obs).all()
    total = 0.0
    for _ in range(300):
        obs, r, _, _, info = env.step(info["expert"])
        total += r
        assert np.isfinite(obs).all() and obs.min() >= -2 and obs.max() <= 2
    assert total > 0  # the heuristic intercepts something in 300 dwells

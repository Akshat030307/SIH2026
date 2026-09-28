import numpy as np
import pytest

from smartscan.sim.emitter import Emitter, Segment, Waveform
from smartscan.sim.engine import RFEngine
from smartscan.sim.propagation import beam_gain_db, noise_power_dbw, snr_db
from smartscan.sim.receiver import ReceiverSpec
from smartscan.sim.world import World, build_world, list_scenarios


def circ_emitter(period=4.0, phase=1.0, pri=1e-3, jitter=0.0, rf=9.3e9):
    wf = Waveform(intervals=[pri], pw=1e-6, rf=[rf], jitter=jitter)
    return Emitter(id=0, name="e", cls="coastal", priority=1.0, bearing_deg=45.0, range_m=50e3,
                   pt_w=50e3, gain_db=33.0, beamwidth_deg=2.0, sidelobe_db=-45.0, scan="circular",
                   segments=[Segment(0.0, 30.0, 0, wf, phase=0.0, seed=3)], scan_period=period, scan_phase=phase)


def test_snr_matches_radar_equation():
    # hand computation of Pt·Gt·Gr·λ²/((4πR)²·kTBFL)
    pt, gt, f, r = 1e6, 35.0, 3e9, 100e3
    lam = 299_792_458.0 / f
    pr = pt * 10 ** (gt / 10) * lam**2 / ((4 * np.pi * r) ** 2 * 10 ** 0.6)
    n = 1.380649e-23 * 290 * 20e6 * 10 ** 1.2
    expected = 10 * np.log10(pr / n)
    got = snr_db(pt, gt, 0.0, f, r, 6.0, noise_power_dbw(290, 20e6, 12.0))
    assert got == pytest.approx(expected, abs=1e-9)


def test_beam_pattern_3db_at_half_beamwidth():
    g = beam_gain_db(np.array([0.0, 1.0, 40.0]), 30.0, 2.0, -40.0)
    assert g[0] == pytest.approx(30.0)
    assert g[1] == pytest.approx(27.0)
    assert g[2] == pytest.approx(-10.0)


def test_illumination_windows_are_periodic():
    e = circ_emitter(period=4.0, phase=1.0)
    w = e.illumination_windows(0, 30, margin_db=3.0)
    centers = (w[:, 0] + w[:, 1]) / 2
    assert np.allclose(np.diff(centers[1:-1]), 4.0)
    assert centers[1] == pytest.approx(5.0)


def test_pulses_deterministic_and_window_additive():
    e = circ_emitter(jitter=0.1)
    a = e.pulses(1.0, 1.1)[0]
    b = e.pulses(1.0, 1.1)[0]
    assert np.array_equal(a, b)
    split = np.concatenate([e.pulses(1.0, 1.05)[0], e.pulses(1.05, 1.1)[0]])
    assert np.allclose(np.sort(split), np.sort(a))
    assert len(a) == pytest.approx(100, abs=2)


def test_engine_detects_only_in_mainlobe_and_books_events():
    e = circ_emitter(period=4.0, phase=1.0)
    rx = ReceiverSpec(false_pulse_rate=0.0)
    world = World("t", 12.0, rx, [e], seed=0)
    eng = RFEngine(world)
    ch = int(rx.channel_of(9.3e9))
    # sit on the channel through the first beam passage (t = 1.0 s)
    eng.t = 0.97
    eng.channel = ch
    r = eng.step(ch, rx.n_dwells - 1)  # 50 ms dwell
    assert len(r.pdws) > 3
    assert np.all(r.pdws["emitter"] == 0)
    assert len(r.new_intercepts) == 1
    # far from the beam: nothing
    eng.step(ch, rx.n_dwells - 1)
    eng.t = 2.5
    r3 = eng.step(ch, rx.n_dwells - 1)
    assert len(r3.pdws) == 0


@pytest.mark.parametrize("name", list_scenarios())
def test_scenarios_build_and_run(name):
    w = build_world(name, seed=0)
    eng = RFEngine(w)
    assert eng.truth.n > 0
    for k in range(50):
        eng.step(k % w.receiver.n_channels, 2)
    assert eng.t > 0

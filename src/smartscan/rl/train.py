"""Train the D3QN scheduler.

    python -m smartscan.rl.train --steps 1000000 --envs 12

Phases:
  1. demonstrations: SmartHeuristic drives all envs; transitions go into the replay
     with the expert action attached (DQfD)
  2. pre-training on demonstrations only (TD + large-margin loss)
  3. online ε-greedy D3QN with demonstrations kept in the replay
Periodically evaluates the greedy policy on held-out seeds, and keeps the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time

import gymnasium as gym
import numpy as np
import torch

from smartscan.rl.behaviour import load_behaviour
from smartscan.rl.d3qn import D3QNAgent, NStep, Replay
from smartscan.rl.env import DEFAULT_SCENARIOS, SmartScanEnv
from smartscan.sim.world import ROOT

CKPT_DIR = ROOT / "checkpoints"


def _make(rank: int, seed: int, scenarios):
    def f():
        return SmartScanEnv(scenarios=scenarios, seed=seed * 1000 + rank, behaviour=load_behaviour())
    return f


def evaluate_policy(agent: D3QNAgent, scenarios, seeds=(500, 501)) -> dict:
    """Greedy-policy validation on seeds disjoint from both training (0-99) and test (100+ in eval.py)."""
    from smartscan.rl.policy import D3QNScheduler
    from smartscan.runner import run_episode
    from smartscan.sim.world import build_world

    sched = D3QNScheduler(net=agent.q)
    rows = []
    agent.q.eval()
    for sc in scenarios:
        for s in seeds:
            m, _, _ = run_episode(build_world(sc, seed=s), sched, seed=s)
            rows.append(m)
    agent.q.train()
    keys = ["pd", "pd_weighted", "intercept_rate", "ttfi_censored_mean", "pfa"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def train(steps=1_000_000, n_envs=12, demo_steps=120_000, pretrain=15_000, batch=256, eps0=0.05, eps1=0.01,
          eval_every=100_000, seed=0, scenarios=DEFAULT_SCENARIOS, replay=400_000, updates_per_step=2):
    from torch.utils.tensorboard import SummaryWriter

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    load_behaviour()  # fit once in the parent before workers start
    run = ROOT / "runs" / time.strftime("d3qn_%Y%m%d_%H%M%S")
    tb = SummaryWriter(run)
    envs = gym.vector.AsyncVectorEnv([_make(i, seed, scenarios) for i in range(n_envs)])
    probe = SmartScanEnv(scenarios=scenarios)
    K, D = probe.K, probe.D
    obs_dim = envs.single_observation_space.shape[0]
    agent = D3QNAgent(K, D, obs_dim)
    rb = Replay(replay, obs_dim)
    rng = np.random.default_rng(seed)
    nsteps = [NStep(agent.n_step, agent.gamma) for _ in range(n_envs)]
    obs, info = envs.reset(seed=seed)
    expert = np.asarray(info["expert"], dtype=np.int64)
    prev_done = np.zeros(n_envs, bool)
    ep_pd, ep_ret, cur_ret = [], [], np.zeros(n_envs)
    best, t0, total, n_upd = -1.0, time.time(), 0, 0
    log = []

    def step_envs(actions, demo):
        nonlocal obs, expert, prev_done
        obs2, r, term, trunc, inf = envs.step(actions)
        done = term | trunc
        for i in range(n_envs):
            if prev_done[i]:
                continue  # autoreset step: obs2[i] is the first obs of a new episode
            cur_ret[i] += r[i]
            for tr in nsteps[i].push(obs[i], actions[i], r[i], obs2[i], done[i], expert[i] if demo else -1):
                rb.add(*tr[:5], a_exp=tr[5], demo=demo)
            if done[i]:
                ep_ret.append(cur_ret[i])
                cur_ret[i] = 0.0
                if "ep_pd_w" in inf and inf["_ep_pd_w"][i]:
                    ep_pd.append(float(inf["ep_pd_w"][i]))
        if "expert" in inf:
            m = inf["_expert"]
            expert = np.where(m, inf["expert"], expert).astype(np.int64)
        prev_done = done
        obs = obs2

    # 1. demonstrations
    print(f"collecting {demo_steps} demonstration steps with {n_envs} envs ...")
    while total < demo_steps:
        step_envs(expert.copy(), demo=True)
        total += n_envs
    print(f"  demos: {rb.n} transitions, heuristic episode pd_w {np.mean(ep_pd) if ep_pd else float('nan'):.3f} "
          f"({time.time() - t0:.0f}s)")
    tb.add_scalar("demo/pd_weighted", np.mean(ep_pd) if ep_pd else 0.0, 0)
    ep_pd.clear()

    # 2. pre-training
    for k in range(pretrain):
        st = agent.update(rb, batch, rng, beta=0.4)
        if k % 1000 == 0:
            print(f"  pretrain {k}: {st}")
            for kk, v in st.items():
                tb.add_scalar(f"pretrain/{kk}", v, k)
    m = evaluate_policy(agent, scenarios)
    print(f"after pretraining: {m}")
    tb.add_scalars("eval", m, 0)
    torch.save({"q": agent.q.state_dict(), "K": K, "D": D, "meta": {"eval": m, "steps": 0}}, CKPT_DIR / "d3qn_pretrained.pt")
    best = m["pd_weighted"]
    agent.save(CKPT_DIR / "d3qn_best.pt", {"eval": m, "steps": 0})

    # 3. online
    online = 0
    next_eval = eval_every
    while online < steps:
        eps = eps0 + (eps1 - eps0) * min(online / (0.5 * steps), 1.0)
        a = agent.act(obs, eps, rng)
        step_envs(a, demo=False)
        online += n_envs
        for _ in range(updates_per_step):
            st = agent.update(rb, batch, rng, beta=0.4 + 0.6 * online / steps)
            n_upd += 1
        if n_upd % 500 == 0:
            for kk, v in st.items():
                tb.add_scalar(f"train/{kk}", v, online)
            tb.add_scalar("train/eps", eps, online)
            if ep_ret:
                tb.add_scalar("train/episode_return", float(np.mean(ep_ret[-20:])), online)
            if ep_pd:
                tb.add_scalar("train/episode_pd_weighted", float(np.mean(ep_pd[-20:])), online)
        if online >= next_eval:
            next_eval += eval_every
            m = evaluate_policy(agent, scenarios)
            tb.add_scalars("eval", m, online)
            rec = {"steps": online, "eps": eps, "train_return": float(np.mean(ep_ret[-20:])) if ep_ret else None,
                   "train_pd_w": float(np.mean(ep_pd[-20:])) if ep_pd else None, **m,
                   "elapsed_s": time.time() - t0}
            log.append(rec)
            print(json.dumps(rec))
            agent.save(CKPT_DIR / "d3qn_last.pt", {"eval": m, "steps": online})
            if m["pd_weighted"] > best:
                best = m["pd_weighted"]
                agent.save(CKPT_DIR / "d3qn_best.pt", {"eval": m, "steps": online})
                print(f"  new best {best:.3f}")
    (CKPT_DIR / "d3qn_train_log.json").write_text(json.dumps(
        {"episode_returns": [float(x) for x in ep_ret], "evals": log}, indent=1))
    envs.close()
    tb.close()
    return agent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--demo-steps", type=int, default=120_000)
    ap.add_argument("--pretrain", type=int, default=15_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    train(a.steps, a.envs, a.demo_steps, a.pretrain, eval_every=a.eval_every, seed=a.seed)


if __name__ == "__main__":
    main()

"""Three call styles, one operation, identical results.

Same BalancedSelector applied to a windowed pipeline three different ways.
Config-parity is a first-class invariant: if the three styles ever diverge,
kuration-sdk-wrapper's config-parity guarantee is broken.
"""
from kuration_sdk_wrapper import BalancedSelector, input, run

sessions = {
    f"session-{i}": [
        {"t": float(step), "signal": float(i * step % 7)}
        for step in range(100)
    ]
    for i in range(3)
}

base = input(sessions, time="t", hz=10.0).window(8, stride=4)

# ---- 1. Direct apply (Tier 1) ----------------------------------------------
op = BalancedSelector(feature="activity", of="signal", bins=4, count=20)
r1 = op.apply(base).options(mode="strict", seed=1).collect()

# ---- 2. run() with an instance (Tier 2) ------------------------------------
r2 = run(base, [
    BalancedSelector(feature="activity", of="signal", bins=4, count=20),
]).options(mode="strict", seed=1).collect()

# ---- 3. run() from a config dict (Tier 2 declarative) ----------------------
config = {
    "operations": [
        {"type": "BalancedSelector", "feature": "activity", "of": "signal",
         "bins": 4, "count": 20},
    ]
}
r3 = run(base, config["operations"]).options(mode="strict", seed=1).collect()

print(f"direct apply:   {len(r1.selected)} windows")
print(f"run(instances): {len(r2.selected)} windows")
print(f"run(config):    {len(r3.selected)} windows")

assert len(r1.selected) == len(r2.selected) == len(r3.selected), (
    "config-parity broken: the three call styles produced different counts"
)
print("OK: three call styles produce identical selection counts")

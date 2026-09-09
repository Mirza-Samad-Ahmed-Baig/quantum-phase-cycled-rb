"""
IBM Quantum connection, backend selection, and calibration logging.

Nothing here requires credentials at import time, so the rest of the project (and every
self-test) runs offline. Credentials are only touched when you actually call `get_service`.

ONE-TIME SETUP (the only manual step in this project)
-----------------------------------------------------
1. Create a free account at https://quantum.cloud.ibm.com/ and copy your API token.
2. Save it once:
       .venv\\Scripts\\python.exe -m hardware.backend --save-token YOUR_TOKEN_HERE
   The token is stored by Qiskit in ~/.qiskit/ and is NOT written into this repo
   (.gitignore excludes tokens and .qiskit/).
3. Verify:
       .venv\\Scripts\\python.exe -m hardware.backend --list

Why calibration logging matters here: the gap scan is sensitive to low-frequency drift, so
every submitted batch records T1, T2, gate/readout errors and a timestamp for the qubit
used. Those snapshots are what let the analysis separate a genuine correlation signal from
a device that simply recalibrated mid-experiment.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# TLS trust: use the OS certificate store.
#
# This network presents an intercepting TLS certificate whose root CA is trusted by
# Windows but is NOT in certifi's bundle. `requests` (used by the IBM SDK) verifies
# against certifi, so every IBM call died with
#     SSLCertVerificationError: unable to get local issuer certificate
# while plain urllib (which uses the Windows store) worked fine -- a confusing
# split that surfaces as a bogus "invalid API token" error from qiskit-ibm-runtime.
#
# `truststore` routes verification through the OS trust store. Verification stays
# fully ENABLED; we are correcting the trust anchors, never disabling the check.
# Never "fix" this with verify=False -- that would send the API token over a
# connection nobody authenticated.
# --------------------------------------------------------------------------- #
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
    TRUSTSTORE_ACTIVE = True
except Exception:  # pragma: no cover - optional dependency
    TRUSTSTORE_ACTIVE = False

__all__ = [
    "save_credentials",
    "credentials_available",
    "get_service",
    "list_backends",
    "pick_backend",
    "backend_dt",
    "calibration_snapshot",
]

# IBM renamed the channel; try the current one first and fall back for older setups.
_CHANNELS = ("ibm_quantum_platform", "ibm_cloud")


def save_credentials(token: str, channel: str = "ibm_quantum_platform", overwrite: bool = True,
                     name: str | None = None):
    """Persist an IBM Quantum API token via Qiskit's account store (never into this repo).

    Pass `name` to store the token as a NAMED account instead of the default. This matters
    when a project has more than one funding source of QPU minutes (e.g. a co-author's
    account): saving unnamed overwrites the default entry, and because the store never hands
    a token back in plaintext, the overwritten one is simply gone. Named accounts are
    additive, so several tokens coexist and `--account NAME` selects between them.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    kw = {"token": token, "channel": channel, "overwrite": overwrite}
    if name:
        kw["name"] = name
    QiskitRuntimeService.save_account(**kw)
    return True


def _account_name(explicit: str | None = None) -> str | None:
    """Which saved account to use: explicit argument, else $QGAP_IBM_ACCOUNT, else default."""
    import os

    return explicit or os.environ.get("QGAP_IBM_ACCOUNT") or None


def credentials_available(name: str | None = None) -> bool:
    """True if a saved IBM account can be loaded, without raising."""
    try:
        get_service(name=name)
        return True
    except Exception:
        return False


def get_service(name: str | None = None):
    """Return a QiskitRuntimeService, with an actionable message if it is not set up.

    `name` selects a named account from the Qiskit store (see `save_credentials`); it falls
    back to $QGAP_IBM_ACCOUNT and then to the default account, so existing call sites that
    pass nothing keep their old behaviour.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    acct = _account_name(name)
    last = None
    if acct:
        # An explicitly requested account must not silently fall back to the default one:
        # submitting a job against the wrong account would spend the wrong minute budget.
        try:
            return QiskitRuntimeService(name=acct)
        except Exception as e:
            raise RuntimeError(
                f"Could not connect using saved IBM account {acct!r}. Check token, instance access and network. Setup:\n"
                f"  .venv\\Scripts\\python.exe -m hardware.backend --save-token TOKEN "
                f"--account {acct}\n(underlying error: {e})"
            ) from e
    try:
        return QiskitRuntimeService()
    except Exception as e:
        last = e
    original_error = last
    for ch in _CHANNELS:
        try:
            return QiskitRuntimeService(channel=ch)
        except Exception as e:
            last = e
    raise RuntimeError(
        "Could not connect using the default IBM account. This may be an instance-access, "
        "token, or network problem; it does not necessarily mean no credentials are saved. "
        "Select a working named account with --account NAME. "
        f"Original connection error: {original_error}; final channel error: {last}"
    )


def list_backends(service=None) -> list[dict]:
    """Summarize available backends: name, qubit count, queue depth, simulator flag."""
    service = get_service() if service is None else service
    out = []
    for b in service.backends():
        try:
            status = b.status()
            out.append({
                "name": b.name,
                "n_qubits": getattr(b, "num_qubits", None),
                "operational": getattr(status, "operational", None),
                "pending_jobs": getattr(status, "pending_jobs", None),
                "simulator": getattr(getattr(b, "configuration", lambda: None)(),
                                     "simulator", False),
            })
        except Exception as e:
            out.append({"name": getattr(b, "name", "?"), "error": str(e)})
    return out


def pick_backend(service=None, name: str | None = None, min_qubits: int = 1):
    """Pick a real (non-simulator) backend: the named one, else the least busy operational."""
    service = get_service() if service is None else service
    if name:
        return service.backend(name)
    try:
        return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    except Exception:
        candidates = [b for b in service.backends(operational=True, simulator=False)
                      if getattr(b, "num_qubits", 0) >= min_qubits]
        if not candidates:
            raise RuntimeError("no operational hardware backend available")
        return sorted(candidates, key=lambda b: b.status().pending_jobs)[0]


def backend_dt(backend) -> float | None:
    """Backend sample time dt in seconds, or None if unavailable (e.g. plain simulator)."""
    for probe in (
        lambda: backend.target.dt,
        lambda: backend.configuration().dt,
        lambda: backend.dt,
    ):
        try:
            v = probe()
            if v:
                return float(v)
        except Exception:
            continue
    return None


def backend_timing(backend) -> dict:
    """Read the device's REAL timing constraints: dt, granularity, min_length, alignments.

    Do not hardcode these. Assuming the common granularity of 16 on a device that actually
    reports 1 quantizes the idle gap 16x too coarsely: on ibm_fez (dt = 4 ns) it forced a
    64 ns minimum step, which collapsed requested gaps of 20/40/80 ns onto a single value
    and left the log-log nu regression with duplicated x-points.
    """
    out = {"dt_s": backend_dt(backend), "granularity": 1, "min_length": 1,
           "pulse_alignment": 1, "acquire_alignment": 1}
    tc = None
    for probe in (
        lambda: backend.configuration().timing_constraints,
        lambda: {
            "granularity": backend.target.granularity,
            "min_length": backend.target.min_length,
            "pulse_alignment": backend.target.pulse_alignment,
            "acquire_alignment": backend.target.acquire_alignment,
        },
    ):
        try:
            tc = probe()
            if tc:
                break
        except Exception:
            continue
    if isinstance(tc, dict):
        for k in ("granularity", "min_length", "pulse_alignment", "acquire_alignment"):
            if tc.get(k):
                out[k] = int(tc[k])
    return out


def calibration_snapshot(backend, qubit: int = 0) -> dict:
    """Record the device state for one qubit: T1, T2, frequency, gate/readout errors, dt.

    Returns whatever the backend exposes; missing fields come back as None rather than
    raising, so a snapshot is always recorded alongside the data.
    """
    snap = {
        "backend": getattr(backend, "name", str(backend)),
        "qubit": int(qubit),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": backend_dt(backend),
    }
    # Legacy properties API (T1/T2/readout/gate errors).
    try:
        props = backend.properties()
        if props is not None:
            for key, fn in (("T1_s", props.t1), ("T2_s", props.t2),
                            ("frequency_Hz", props.frequency)):
                try:
                    snap[key] = float(fn(qubit))
                except Exception:
                    snap[key] = None
            try:
                snap["readout_error"] = float(props.readout_error(qubit))
            except Exception:
                snap["readout_error"] = None
            for g in ("sx", "x", "rz", "id"):
                try:
                    snap[f"gate_error_{g}"] = float(props.gate_error(g, qubit))
                except Exception:
                    pass
            try:
                snap["last_update_date"] = str(props.last_update_date)
            except Exception:
                pass
    except Exception as e:
        snap["properties_error"] = str(e)

    # Modern Target API as a fallback / supplement.
    try:
        target = backend.target
        for g in ("sx", "x"):
            try:
                inst = target[g][(qubit,)]
                snap.setdefault(f"gate_error_{g}", None)
                if snap.get(f"gate_error_{g}") is None and inst.error is not None:
                    snap[f"gate_error_{g}"] = float(inst.error)
                if inst.duration is not None:
                    snap[f"gate_duration_{g}_s"] = float(inst.duration)
            except Exception:
                continue
    except Exception:
        pass
    return snap


def append_calibration_log(snapshot: dict, path: str | Path = "data/calibration_log.jsonl") -> Path:
    """Append a calibration snapshot to a JSONL log (one line per submitted batch)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")
    return p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IBM Quantum setup and inspection")
    ap.add_argument("--save-token", metavar="TOKEN", help="save an IBM Quantum API token")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--account", metavar="NAME", default=None,
                    help="named account in the Qiskit store (also settable via "
                         "$QGAP_IBM_ACCOUNT); omit to use the default account")
    ap.add_argument("--accounts", action="store_true",
                    help="list saved account names (never prints tokens)")
    ap.add_argument("--list", action="store_true", help="list available backends")
    ap.add_argument("--pick", action="store_true", help="show the least-busy real backend")
    ap.add_argument("--calibration", metavar="BACKEND", nargs="?", const="",
                    help="dump a calibration snapshot")
    ap.add_argument("--qubit", type=int, default=0)
    args = ap.parse_args()

    did_something = False

    if args.accounts:
        from qiskit_ibm_runtime import QiskitRuntimeService

        saved = QiskitRuntimeService.saved_accounts()
        print(f"{'account':<20} {'channel':<24} instance")
        for nm, info in saved.items():
            print(f"{nm:<20} {str(info.get('channel')):<24} {info.get('instance') or '-'}")
        if not saved:
            print("(none saved)")
        did_something = True

    if args.save_token:
        save_credentials(args.save_token, channel=args.channel, name=args.account)
        where = f"account {args.account!r}" if args.account else "the DEFAULT account"
        print(f"Token saved to {where} (channel={args.channel}). Verify with: "
              f"python -m hardware.backend --list"
              + (f" --account {args.account}" if args.account else ""))
        did_something = True

    if args.list or args.pick or args.calibration is not None:
        if not credentials_available(args.account):
            print("No IBM Quantum credentials found.")
            print("  1) Sign up free at https://quantum.cloud.ibm.com/ and copy your API token")
            print("  2) .venv\\Scripts\\python.exe -m hardware.backend --save-token YOUR_TOKEN")
            raise SystemExit(1)
        service = get_service(args.account)

        if args.list:
            print(f"{'backend':<28} {'qubits':>7} {'queue':>7}  operational")
            for b in list_backends(service):
                if "error" in b:
                    print(f"{b['name']:<28} {'?':>7} {'?':>7}  ERROR: {b['error']}")
                else:
                    print(f"{b['name']:<28} {str(b['n_qubits']):>7} "
                          f"{str(b['pending_jobs']):>7}  {b['operational']}")
            did_something = True

        if args.pick:
            b = pick_backend(service)
            print(f"least busy real backend: {b.name} "
                  f"({getattr(b, 'num_qubits', '?')} qubits, dt={backend_dt(b)})")
            did_something = True

        if args.calibration is not None:
            b = pick_backend(service, name=args.calibration or None)
            snap = calibration_snapshot(b, args.qubit)
            print(json.dumps(snap, indent=2))
            did_something = True

    if not did_something:
        ap.print_help()

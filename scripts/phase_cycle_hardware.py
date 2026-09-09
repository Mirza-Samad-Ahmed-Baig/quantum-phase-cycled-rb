"""Small injected phase-cycle control on IBM, with durable counts and offline analysis.

Prepare is offline. Submit checks the saved allowance and caps execution time.
All four arms use the same random Clifford pairs; settings are randomized in
execution order. Positive control is deliberately circuit-constant +/- angle
noise, not naturally occurring device memory. Readout calibration is resampled.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim.phase_cycle import SETTINGS, explicit_survival
from estimator.phase_cycle import block_components, connected_estimate

ARMS=("correlated","reset","static","none")
DEFAULT_PLAN="data/phase_cycle_design.json"


def prepare(a):
    if a.blocks<2 or a.shots<1 or a.delta_ns<=0 or a.max_seconds<=0 or not np.isfinite(a.angle):
        raise ValueError("require >=2 blocks, positive shots/delay/budget and a finite phase")
    rng=np.random.default_rng(a.seed)
    seq=rng.integers(0,24,(a.blocks,2))
    signs=rng.choice([-1,1],(a.blocks,2))
    phases=np.stack((np.repeat(signs[:,:1],2,axis=1)*a.angle,
                     signs*a.angle,np.full((a.blocks,2),a.angle),np.zeros((a.blocks,2))))
    design=dict(protocol="phase-cycle-v1",kind="engineered_noise_control",
                created_utc=datetime.now(timezone.utc).isoformat(),seed=a.seed,
                backend=a.backend,qubit=a.qubit,blocks=a.blocks,shots=a.shots,
                angle=a.angle,beta=float(np.pi/2),delta_ns=a.delta_ns,
                settings=list(SETTINGS),arms=list(ARMS),sequences=seq.tolist(),phases=phases.tolist(),
                calibration_repeats=64,maximum_qpu_seconds=a.max_seconds,
                packed=bool(a.packed),
                decision="two-sided paired block bootstrap; correlated-minus-reset contrast is primary",
                primary_truth=float(4/9*np.sin(a.angle)**2),
                caveats=["Only injected noise is paired across settings; intrinsic noise is not replayed.",
                         "Affine readout model and stable gate errors are assumptions; controls test their impact.",
                         "This is not a quantum-memory certification or a native-memory detection."])
    design["circuits"]=len(ARMS)*a.blocks*len(SETTINGS)+2*design["calibration_repeats"]
    design["circuit_shots"]=design["circuits"]*a.shots
    # Historical campaign used about 0.00030 seconds per circuit shot. Include
    # 10 seconds overhead; refuse to submit if this budget does not fit.
    design["estimated_qpu_seconds"]=round(design["circuit_shots"]*.00030+10,2)
    Path(a.plan).write_text(json.dumps(design,indent=2),encoding="utf-8")
    print({k:design[k] for k in ("circuits","circuit_shots","estimated_qpu_seconds","primary_truth")},flush=True)
    return design


def build(design,dt=None,granularity=16):
    from qiskit import QuantumCircuit
    from hardware.circuits import build_gapscan_circuit
    from sim.clifford import clifford_words, compose_index_table, inverse_index_table
    words,table,inv=clifford_words(),compose_index_table(),inverse_index_table()
    seq=np.asarray(design["sequences"])
    phases=np.asarray(design["phases"])
    beta=design["beta"]
    circuits,meta=[],[]
    for ai,arm in enumerate(ARMS):
        for block in range(design["blocks"]):
            for si,setting in enumerate(SETTINGS):
                if si<4:
                    sign=np.array(((1,1),(1,-1),(-1,1),(-1,-1))[si])
                    idx=seq[block]
                    theta=phases[ai,block]+beta*sign
                else:
                    slot=(si-4)//2
                    sign=1 if si%2==0 else -1
                    idx=seq[block,slot:slot+1]
                    theta=phases[ai,block,slot:slot+1]+beta*sign
                qc=build_gapscan_circuit(idx,design["delta_ns"]*1e-9,
                     inject_phases=theta,dt=dt,granularity=granularity,
                     words=words,table=table,inv=inv)
                circuits.append(qc)
                meta.append(dict(arm=arm,block=block,setting=setting))
    for repeat in range(design["calibration_repeats"]):
        for state in (0,1):
            qc=QuantumCircuit(1,1)
            if state: qc.x(0)
            qc.measure(0,0)
            circuits.append(qc)
            meta.append(dict(calibration_state=state,repeat=repeat))
    order=np.random.default_rng(design["seed"]+101).permutation(len(circuits))
    return [circuits[i] for i in order],[meta[i] for i in order]


def check_design(a):
    """Independent circuit propagation and affine-readout false-positive check."""
    from qiskit.quantum_info import Statevector
    import copy
    original=json.loads(Path(a.plan).read_text(encoding="utf-8"))
    d=copy.deepcopy(original)
    d["blocks"]=2;d["sequences"]=d["sequences"][:2]
    d["phases"]=[arm[:2] for arm in d["phases"]]
    d["calibration_repeats"]=0
    circuits,meta=build(d)
    seq,phases=np.asarray(d["sequences"]),np.asarray(d["phases"])
    errors=[]
    for circuit,mt in zip(circuits,meta):
        ai,block,si=ARMS.index(mt["arm"]),mt["block"],SETTINGS.index(mt["setting"])
        if si<4:
            idx=seq[block]
            theta=phases[ai,block]+d["beta"]*np.array(((1,1),(1,-1),(-1,1),(-1,-1))[si])
        else:
            slot=(si-4)//2
            idx=seq[block,slot:slot+1]
            theta=phases[ai,block,slot:slot+1]+(1 if si%2==0 else -1)*d["beta"]
        ideal=explicit_survival(idx[None,:],theta[None,:])[0]
        actual=Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()[0]
        errors.append(abs(actual-ideal))
    assert max(errors)<1e-12
    # Deterministic detuning has nonzero marginal odd responses. Its connected
    # moment must stay zero after a nontrivial affine readout transformation.
    theta=original["angle"];beta=original["beta"]
    twirl=lambda x:(1+2*np.cos(x))/3
    pp,pm=twirl(theta+beta),twirl(theta-beta)
    ideal=np.array([pp*pp,pp*pm,pm*pp,pm*pm,pp,pm,pp,pm])/2+.5
    gain,offset=.87,.08
    observed=gain*ideal+offset
    c=block_components(np.tile(observed,(10,1)))
    naive=connected_estimate(c)
    corrected=connected_estimate(c/gain)
    assert abs(corrected)<1e-12 and abs(naive)>.001
    result=dict(plan=a.plan,plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),
                circuits_checked=len(circuits),max_statevector_error=float(max(errors)),
                affine_static_naive_contrast=naive,affine_static_corrected_contrast=corrected)
    out=Path(a.plan).with_name(Path(a.plan).stem+"_check.json")
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


def pack_circuits(circuits,meta,design):
    """Broadcast phase bindings through shared Clifford templates.

    This reduces thousands of separate Runtime PUBs to a few hundred templates,
    without changing any circuit, shot count, phase or control. The inverse
    remains independent of the phases. Metadata retains every binding's identity.
    """
    from qiskit.circuit import ParameterVector
    seq=np.asarray(design["sequences"])
    groups={}
    for qc,mt in zip(circuits,meta,strict=True):
        if "arm" not in mt:
            groups[("cal",mt["calibration_state"],mt["repeat"])]=[qc,None,[mt]]
            continue
        si=SETTINGS.index(mt["setting"])
        indices=seq[mt["block"]] if si<4 else seq[mt["block"],(si-4)//2:(si-4)//2+1]
        key=tuple(map(int,indices))
        angles=[float(inst.operation.params[0]) for inst in qc.data if inst.operation.name=="rz"]
        if len(angles)!=len(indices):raise ValueError("unexpected Rz outside injected slots")
        if key not in groups:
            template=qc.copy_empty_like()
            parameters=ParameterVector("phase",len(indices))
            k=0
            for inst in qc.data:
                if inst.operation.name=="rz":
                    template.rz(parameters[k],0);k+=1
                else:template.append(inst.operation,inst.qubits,inst.clbits)
            groups[key]=[template,[],[]]
        groups[key][1].append(angles)
        groups[key][2].append(mt)
    items=list(groups.values())
    np.random.default_rng(design["seed"]+102).shuffle(items)
    return ([x[0] for x in items],[x[1] for x in items],[x[2] for x in items])


def submit(a):
    from hardware import backend as hb
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    design=json.loads(Path(a.plan).read_text(encoding="utf-8"))
    svc=hb.get_service(a.account)
    usage=svc.usage()
    remaining=float(usage["usage_remaining_seconds"])
    cap=min(float(design["maximum_qpu_seconds"]),remaining-5)
    if design["estimated_qpu_seconds"]>cap:
        raise RuntimeError(f"Estimated {design['estimated_qpu_seconds']} seconds exceeds cap {cap}; smaller design needed")
    backend=svc.backend(design["backend"])
    dt=hb.backend_dt(backend)
    circuits,meta=build(design,dt=dt)
    if design.get("packed"):
        circuits,bindings,pub_meta=pack_circuits(circuits,meta,design)
        meta=[mt for group in pub_meta for mt in group]
    else:
        bindings=[None]*len(circuits);pub_meta=[[mt] for mt in meta]
    print(f"Transpiling {len(circuits)} PUB templates, {len(meta)} evaluations; remaining {remaining:.0f}s, execution cap {cap:.0f}s",flush=True)
    tqc=transpile(circuits,backend=backend,initial_layout=[design["qubit"]],optimization_level=1,num_processes=1)
    for c,group in zip(tqc,pub_meta):
        if "arm" in group[0] and c.count_ops().get("delay",0)<2:
            raise RuntimeError("transpilation lost the split idle windows")
    # No account token or instance credential is serialized.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pending=dict(design=design,plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),
                 execution_meta=meta,usage_before=usage,calibration=hb.calibration_snapshot(backend,design["qubit"]),
                 timestamp_utc=stamp,dt=dt,max_execution_time=cap,
                 pub_sizes=[len(group) for group in pub_meta])
    path=Path(f"data/phase_cycle_{stamp}_pending.json")
    path.write_text(json.dumps(pending,indent=2,default=str),encoding="utf-8")
    sampler=SamplerV2(mode=backend,options={"max_execution_time":int(cap)})
    pubs=[c if values is None else (c,np.asarray(values)) for c,values in zip(tqc,bindings)]
    job=sampler.run(pubs,shots=design["shots"])
    pending["job_id"]=job.job_id()
    path.write_text(json.dumps(pending,indent=2,default=str),encoding="utf-8")
    print(f"SUBMITTED {job.job_id()} pending={path}",flush=True)


def fetch(a):
    from hardware import backend as hb
    from hardware.run_gapscan import _counts_from_job_result
    p=Path(a.fetch)
    pending=json.loads(p.read_text(encoding="utf-8"))
    job=hb.get_service(a.account).job(pending["job_id"])
    print("job status:",job.status(),flush=True)
    if str(job.status()).upper()!="DONE":
        if str(job.status()).upper() in ("ERROR","CANCELLED"):
            failure=dict(job_id=pending["job_id"],status=str(job.status()),
                         error=job.error_message(),metrics=job.metrics())
            fp=p.with_name(p.name.replace("_pending.json","_failure.json"))
            fp.write_text(json.dumps(failure,indent=2,default=str),encoding="utf-8")
            print(json.dumps(failure,indent=2,default=str),flush=True)
        return
    result=job.result()
    if pending.get("pub_sizes"):
        counts=[]
        for pub,size in zip(result,pending["pub_sizes"],strict=True):
            bits=next(iter(pub.data.values()))
            if bits.shape:
                if int(np.prod(bits.shape))!=size:raise ValueError("PUB result shape mismatch")
                counts.extend(bits.get_counts(index) for index in np.ndindex(bits.shape))
            else:
                if size!=1:raise ValueError("scalar PUB unexpectedly has multiple metadata rows")
                counts.append(bits.get_counts())
    else:counts=_counts_from_job_result(result)
    if len(counts)!=len(pending["execution_meta"]):
        raise RuntimeError("count/metadata length mismatch")
    raw=dict(job_id=pending["job_id"],pending_file=str(p),counts=counts,metrics=job.metrics())
    path=p.with_name(p.name.replace("_pending.json","_counts.json"))
    path.write_text(json.dumps(raw,indent=2,default=str),encoding="utf-8")
    print("Saved raw counts",path,flush=True)
    analyse(path,a.bootstrap)


def analyse(path,n_boot=4000):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    pending=json.loads(Path(raw["pending_file"].replace("\\", "/")).read_text(encoding="utf-8"))
    d=pending["design"]
    p=np.full((len(ARMS),d["blocks"],len(SETTINGS)),np.nan)
    cal_success=np.zeros(2,dtype=int);cal_shots=np.zeros(2,dtype=int)
    for counts,meta in zip(raw["counts"],pending["execution_meta"],strict=True):
        shots=sum(counts.values())
        if shots!=d["shots"]: raise ValueError("actual shots differ from saved design")
        zero=counts.get("0",0)
        if "calibration_state" in meta:
            k=meta["calibration_state"];cal_success[k]+=zero;cal_shots[k]+=shots
        else:
            p[ARMS.index(meta["arm"]),meta["block"],SETTINGS.index(meta["setting"])]=zero/shots
    if not np.isfinite(p).all(): raise ValueError("missing circuit observations")
    cp=cal_success/cal_shots
    gain=float(cp[0]-cp[1])
    if gain<.5: raise ValueError("readout contrast too small for this protocol")
    components=np.array([block_components(x) for x in p])
    def estimate(c,g):
        return connected_estimate(c/g)
    # All three components are divided by gain; the connected subtraction then
    # has the correct single gain for the joint term and squared gain for means.
    point=np.array([estimate(c,gain) for c in components])
    rng=np.random.default_rng(874302)
    boots=[]
    for _ in range(n_boot):
        idx=rng.integers(0,d["blocks"],d["blocks"])
        cb=rng.binomial(cal_shots,cp)/cal_shots
        gb=cb[0]-cb[1]
        if gb<=0: raise ValueError("bootstrap readout gain is nonpositive")
        boots.append([estimate(c[idx],gb) for c in components])
    boots=np.asarray(boots)
    rows=[]
    for ai,arm in enumerate(ARMS):
        ci=np.quantile(boots[:,ai],[.025,.975])
        rows.append(dict(arm=arm,connected=float(point[ai]),ci=ci.tolist(),
                         stderr=float(boots[:,ai].std(ddof=1)),
                         ideal_ensemble_truth=d["primary_truth"] if arm=="correlated" else 0.))
    diff=boots[:,0]-boots[:,1]
    primary=dict(contrast="correlated minus reset",estimate=float(point[0]-point[1]),
                 ci=np.quantile(diff,[.025,.975]).tolist(),stderr=float(diff.std(ddof=1)))
    result=dict(kind="hardware_injected_control",job_id=raw["job_id"],backend=d["backend"],qubit=d["qubit"],
                blocks=d["blocks"],shots=d["shots"],angle=d["angle"],beta=d["beta"],
                readout_gain=gain,calibration_success=cal_success.tolist(),calibration_shots=cal_shots.tolist(),
                bootstrap=n_boot,arms=rows,primary=primary,metrics=raw["metrics"],
                caveats=d["caveats"],raw_counts_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    out=Path(path).with_name(Path(path).name.replace("_counts.json","_results.json"))
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--submit",action="store_true")
    mode.add_argument("--fetch",metavar="PENDING_JSON")
    mode.add_argument("--analyse",metavar="COUNTS_JSON")
    ap.add_argument("--plan",default=DEFAULT_PLAN)
    ap.add_argument("--account",default="coauthor")
    ap.add_argument("--backend",default="ibm_fez")
    ap.add_argument("--qubit",type=int,default=46)
    ap.add_argument("--blocks",type=int,default=256)
    ap.add_argument("--shots",type=int,default=16)
    ap.add_argument("--angle",type=float,default=.8)
    ap.add_argument("--delta-ns",type=float,default=320.)
    ap.add_argument("--seed",type=int,default=2026090501)
    ap.add_argument("--max-seconds",type=int,default=65)
    ap.add_argument("--bootstrap",type=int,default=4000)
    ap.add_argument("--packed",action="store_true",help="broadcast phase bindings through shared Clifford templates")
    a=ap.parse_args()
    if a.prepare: prepare(a)
    elif a.check: check_design(a)
    elif a.submit: submit(a)
    elif a.fetch: fetch(a)
    else: analyse(a.analyse,a.bootstrap)


if __name__=="__main__": main()

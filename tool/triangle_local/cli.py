import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .exact import numerical_model, rationalize, verify, fraction
from .families import (FAMILIES, family, note_model, ring_half_model,
                       ring_endpoint_model, parity_model, fritz_model)
from .model import lift_model
from .io import distribution_document, dump_json, from_document, load_distribution, load_json
from .optimize import fit
from .serialization import public_json


def bundle(model, target, *, max_denominator=10000, tolerance="1/1000000", search=None):
    rational = rationalize(model, max_denominator)
    return {"schema": "triangle-local-bundle/v1", "target": distribution_document(target),
            "model": rational, "certificate": verify(rational, target, tolerance=tolerance),
            "search": search}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finite triangle-local models with exact rational checking")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Write a benchmark target")
    gen.add_argument("family", choices=FAMILIES)
    gen.add_argument("--parameter", default="1", help="Visibility, u² for renou, or t=v/sqrt(2) for fritz")
    gen.add_argument("--u-squared", default="17/20", help="Renou shape parameter in noisy scans")
    gen.add_argument("--out", required=True)
    solve = sub.add_parser("solve", help="Search, rationalize, and certify a candidate")
    solve.add_argument("input")
    solve.add_argument("--sizes", nargs=3, type=int, default=[4,4,4], metavar=("ALPHA","BETA","GAMMA"))
    solve.add_argument("--outputs", nargs=3, type=int, help="CSV output alphabet sizes, including unused labels")
    solve.add_argument("--symmetry", choices=["none","exchangeable","outcome-equivariant",
                                             "color-equivariant","pair-equivariant"], default="none")
    solve.add_argument("--method", choices=["lbfgs","adam","alternating","em","convex",
                                           "deterministic","least-squares","hybrid","neural"], default="hybrid")
    solve.add_argument("--loss", choices=["l1","l2","kl"], default="l2")
    solve.add_argument("--starts", type=int, default=3)
    solve.add_argument("--steps", type=int, default=500)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--learning-rate", type=float, default=.08)
    solve.add_argument("--max-denominator", type=int, default=10000)
    solve.add_argument("--tolerance", default="1/1000000", help="Exact L1 certificate tolerance")
    solve.add_argument("--normalize-input", action="store_true")
    solve.add_argument("--round", action="store_true", help="Also certify an argmax-rounded candidate")
    solve.add_argument("--initial", help="Warm start from a rational model or result bundle")
    solve.add_argument("--seed-model", choices=["note","ring-half","ring-endpoint","parity","fritz"],
                       help="Lift a known local construction to the requested source sizes")
    solve.add_argument("--jitter", type=float, default=0., help="Response perturbation after lifting a seed")
    solve.add_argument("--hops", type=int, default=2, help="Additional hybrid perturbation searches")
    solve.add_argument("--polish-steps", type=int, default=120, help="Hybrid convex polishing sweeps")
    solve.add_argument("--inner-steps", type=int, default=200, help="L2 block projected-gradient cap")
    solve.add_argument("--optimizer-tolerance", type=float, default=1e-20, help="Numerical objective tolerance")
    solve.add_argument("--fixed-weights", action="store_true", help="Keep weights fixed in convex/deterministic updates")
    solve.add_argument("--out", required=True)
    check = sub.add_parser("verify", help="Recompute and compare a saved certificate")
    check.add_argument("bundle")
    check.add_argument("--input", help="Check against this external target as well as the saved target")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            target = family(args.family, args.parameter, args.u_squared)
            dump_json(args.out, distribution_document(target, family=args.family, parameter=args.parameter,
                       u_squared=args.u_squared,
                       target_arithmetic=("rational; some Renou parameters require a 10^12-denominator approximation"
                                          if args.family.startswith("renou") else "exact rational")))
            print(f"Wrote {args.out}")
        elif args.command == "verify":
            doc = load_json(args.bundle)
            if doc.get("schema") != "triangle-local-bundle/v1":
                raise ValueError("Unsupported bundle schema")
            saved_target = from_document(doc["target"])
            saved = doc["certificate"]
            actual = verify(doc["model"], saved_target, tolerance=saved["L1_tolerance"])
            if actual != saved:
                raise ValueError("Saved certificate differs from independent recomputation")
            if args.input:
                normalized = doc.get("source_input",{}).get("normalization_requested",False)
                external = verify(doc["model"], load_distribution(args.input,outputs=saved_target.shape,normalize=normalized),
                                  tolerance=saved["L1_tolerance"])
                if external != actual:
                    raise ValueError("External target differs from certified target")
            if "rounded" in doc:
                rounded = doc["rounded"]
                if verify(rounded["model"], saved_target, tolerance=saved["L1_tolerance"]) != rounded["certificate"]:
                    raise ValueError("Rounded certificate differs from independent recomputation")
            print(public_json({k: actual[k] for k in ["status","exact_match","L1","L2_squared",
                                                    "model_S3_symmetric_exactly"]}))
        else:
            target = load_distribution(args.input, args.outputs, args.normalize_input)
            fraction(args.tolerance)  # Validate before expensive search.
            if fraction(args.tolerance) < 0 or args.max_denominator < 1:
                raise ValueError("Nonnegative tolerance and positive denominator required")
            initial = None
            if args.initial and args.seed_model:
                raise ValueError("Choose either --initial or --seed-model")
            if args.initial:
                doc = load_json(args.initial)
                initial = numerical_model(doc.get("model", doc))
            if args.seed_model:
                initial = {"note":note_model,"ring-half":ring_half_model,
                           "ring-endpoint":ring_endpoint_model,"parity":parity_model,
                           "fritz":fritz_model}[args.seed_model]()
            if initial is not None and (initial.sizes != tuple(args.sizes) or args.jitter):
                initial = lift_model(initial,tuple(args.sizes),jitter=args.jitter,seed=args.seed)
            result = fit(np.asarray(target,float), tuple(args.sizes), symmetry=args.symmetry,
                         method=args.method, loss=args.loss, starts=args.starts, steps=args.steps,
                         seed=args.seed, learning_rate=args.learning_rate, initial=initial,
                         hops=args.hops,polish_steps=args.polish_steps,inner_steps=args.inner_steps,
                         tolerance=args.optimizer_tolerance,optimize_weights=not args.fixed_weights)
            output = bundle(result.model, target, max_denominator=args.max_denominator,
                            tolerance=args.tolerance, search={**result.summary(), "method":args.method,
                            "seed":args.seed, "sizes":args.sizes, "symmetry":args.symmetry,
                            "input_normalization_requested":args.normalize_input,
                            "starts":args.starts,"steps":args.steps,"hops":args.hops,
                            "polish_steps":args.polish_steps,"inner_steps":args.inner_steps,
                            "optimizer_tolerance":args.optimizer_tolerance,
                            "learning_rate":args.learning_rate,"seed_model":args.seed_model,
                            "initial_file":args.initial,"jitter":args.jitter,
                            "fixed_weights_in_block_updates":args.fixed_weights})
            raw=Path(args.input).read_bytes()
            source={"path":str(args.input),"sha256":sha256(raw).hexdigest(),
                    "normalization_requested":args.normalize_input}
            if Path(args.input).suffix.lower()!=".csv":
                original=json.loads(raw)
                if isinstance(original,dict):
                    source["metadata"]={k:original[k] for k in
                        ("family","parameter","u_squared","target_arithmetic","citation") if k in original}
            output["source_input"]=source
            if args.round:
                rounded = bundle(result.model.rounded(), target, max_denominator=args.max_denominator,
                                 tolerance=args.tolerance)
                output["rounded"] = {k:rounded[k] for k in ("model","certificate")}
            dump_json(args.out, output)
            print(public_json({"output":args.out, "objective":result.objective,
                              "elapsed_seconds":result.elapsed_seconds,
                              "rational_status":output["certificate"]["status"],
                              "rational_L1_decimal":float(fraction(output["certificate"]["L1"])),
                              "rational_L1":output["certificate"]["L1"]}))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

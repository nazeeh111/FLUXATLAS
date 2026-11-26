"""Command-line workflows with explicit reference/prediction provenance."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__
from .config import Experiment


def project_root():
    source=Path(__file__).resolve().parents[2]
    if (source/"web").is_dir():return source
    installed=Path(sys.prefix)/"share/fluxatlas"
    if (installed/"web").is_dir():return installed
    raise RuntimeError("Observatory assets unavailable; use a complete release or source checkout")


def read_experiment(path):
    """Import bare CLI parameters or the existing schema-1 UI configuration export."""
    document=json.loads(path.read_text())
    if not isinstance(document,dict):
        raise ValueError("Configuration JSON must be an object")
    if "schema_version" in document or "config" in document:
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("Unsupported exported configuration schema; expected schema_version 1")
        if set(document) - {"schema_version","config","seed","histories","units","source"}:
            raise ValueError("Unrecognized exported configuration fields")
        config=document.get("config")
        if not isinstance(config,dict) or set(config) != set(Experiment.__dataclass_fields__):
            raise ValueError("Exported configuration requires a complete experiment config object")
        return Experiment(**config)
    return Experiment(**document)


def main():
    parser=argparse.ArgumentParser(prog="fluxatlas",
        description="Inspect saved transport fields and query frozen numeric weights. OpenMC is optional for new references.",
        epilog="Start without a solver: fluxatlas predict --output prediction.npz\n"
               "Check the default experiment: fluxatlas compare --prediction prediction.npz --example-reference\n"
               "The observatory binds to loopback; no browser opens automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version",action="version",version=f"fluxatlas {__version__}")
    sub=parser.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve",help="Serve saved archive and trained-model queries on loopback",
        description="Open the printed URL yourself. Model queries need no OpenMC; new reference calculations require it. Ctrl-C stops the server.")
    serve.add_argument("--port",type=int,default=8765,help="Loopback port (default: 8765; 0 chooses an available port)")
    serve.add_argument("--cache",type=Path,help="Output cache for explicit OpenMC runs; archive replay and model queries do not use it")
    example=sub.add_parser("example",help="Execute a new synthetic OpenMC calculation; no nuclear-data download")
    example.add_argument("--output",type=Path,default=Path("fluxatlas-example"));example.add_argument("--particles",type=int,default=1000)
    example.add_argument("--seed",type=int,default=123457)
    predict=sub.add_parser("predict",help="Query frozen numeric trained weights without OpenMC",
        description="Create a new portable fixed-grid prediction from the bundled weights. Does not train a model or run a solver.")
    predict.add_argument("--config",type=Path,help="Bare experiment JSON or schema-1 UI Export configuration JSON; bare partial inputs keep documented defaults")
    predict.add_argument("--output",type=Path,default=Path("prediction.npz"),help="New numeric NPZ path (default: prediction.npz); existing files are refused")
    compare=sub.add_parser("compare",help="Compare portable prediction with an existing reference",
        description="Check fixed grid, configuration, units and saved provenance before numerical diagnostics. No solver execution. Keep metadata.json beside the reference field.npz.")
    compare.add_argument("--prediction", type=Path, required=True)
    reference = compare.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference", type=Path, help="Saved reference field with adjacent metadata.json")
    reference.add_argument("--example-reference", action="store_true",
                           help="Use the bundled reference for the default experiment")
    compare.add_argument("--output",type=Path,help="Optional new comparison JSON; existing files are refused, stdout still contains the result")
    generate=sub.add_parser("generate",help="Generate the fixed 156-configuration dataset")
    generate.add_argument("--output",type=Path,default=Path("data"));generate.add_argument("--cache",type=Path,default=Path(".cache/fluxatlas"))
    train=sub.add_parser("train",help="Train, calibrate, evaluate and export observatory data")
    train.add_argument("--project",type=Path,default=Path.cwd())
    validate=sub.add_parser("validate",help="Run analytical absorption and repeated-seed convergence")
    validate.add_argument("--output",type=Path,default=Path("benchmarks/physics"));validate.add_argument("--cache",type=Path,default=Path(".cache/fluxatlas"))
    args=parser.parse_args()
    try:
        if args.command=="serve":
            if not 0 <= args.port <= 65535:
                raise ValueError("port must be in 0..65535 (0 selects an available loopback port)")
            from .server import serve
            return serve(project_root(),args.port,args.cache)
        if args.command=="example":
            from .transport import run_reference
            cfg=Experiment();r=run_reference(cfg,args.output,args.particles,20,args.seed,reuse=False)
            print(json.dumps({k:v for k,v in r.items() if k not in {"mean","stderr"}},indent=2));return
        if args.command=="predict":
            from .model import FrozenSurrogate
            from .artifacts import save_prediction
            cfg=read_experiment(args.config) if args.config else Experiment()
            model_path=project_root()/"data/model.npz"
            model=FrozenSurrogate(model_path);pred,half=model.predict(cfg.vector())
            if float(model.state.get("alpha",.1)) != .1:raise ValueError("Prediction export requires the supported 90% interval contract")
            metadata=save_prediction(args.output,cfg,pred.reshape(2,20,24),half.reshape(2,20,24),model.sha256,
                                     model.out_of_domain(cfg.vector(),cfg.family)[0])
            print(json.dumps({**metadata,"output":str(args.output)}));return
        if args.command=="compare":
            from .artifacts import compare_fields,save_comparison
            reference = args.reference
            if args.example_reference:
                reference = project_root() / "examples/reference/018aaf1e873bfacdc6f5/field.npz"
            result = compare_fields(args.prediction, reference)
            if args.output:save_comparison(args.output,result)
            print(json.dumps(result,allow_nan=False,indent=2));return
        if args.command=="generate":
            from .dataset import generate
            return generate(args.output,args.cache)
        if args.command=="train":
            from .study import study
            study(args.project);return
        if args.command=="validate":
            from .validation import validate
            validate(args.output,args.cache);return
    except (RuntimeError,ValueError,TypeError,KeyError,OSError) as exc:
        parser.exit(2,f"fluxatlas: {exc}\n")

if __name__=="__main__":main()

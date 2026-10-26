"""Command-line workflows with explicit reference/prediction provenance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .config import Experiment


def project_root():
    source=Path(__file__).resolve().parents[2]
    if (source/"web").is_dir():return source
    installed=Path(sys.prefix)/"share/fluxatlas"
    if (installed/"web").is_dir():return installed
    raise RuntimeError("Observatory assets unavailable; use a complete release or source checkout")


def main():
    parser=argparse.ArgumentParser(prog="fluxatlas",description="Educational synthetic-data OpenMC transport laboratory")
    parser.add_argument("--version",action="version",version="fluxatlas 0.1.0")
    sub=parser.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve",help="Serve local observatory and validated solver API")
    serve.add_argument("--port",type=int,default=8765);serve.add_argument("--cache",type=Path)
    example=sub.add_parser("example",help="Execute a new synthetic OpenMC calculation; no nuclear-data download")
    example.add_argument("--output",type=Path,default=Path("fluxatlas-example"));example.add_argument("--particles",type=int,default=1000)
    example.add_argument("--seed",type=int,default=123457)
    predict=sub.add_parser("predict",help="Query numeric trained weights without OpenMC")
    predict.add_argument("--config",type=Path);predict.add_argument("--output",type=Path,default=Path("prediction.npz"))
    generate=sub.add_parser("generate",help="Generate the fixed 156-configuration dataset")
    generate.add_argument("--output",type=Path,default=Path("data"));generate.add_argument("--cache",type=Path,default=Path(".cache/fluxatlas"))
    train=sub.add_parser("train",help="Train, calibrate, evaluate and export observatory data")
    train.add_argument("--project",type=Path,default=Path.cwd())
    validate=sub.add_parser("validate",help="Run analytical absorption and repeated-seed convergence")
    validate.add_argument("--output",type=Path,default=Path("benchmarks/physics"));validate.add_argument("--cache",type=Path,default=Path(".cache/fluxatlas"))
    args=parser.parse_args()
    try:
        if args.command=="serve":
            from .server import serve
            return serve(project_root(),args.port,args.cache)
        if args.command=="example":
            from .transport import run_reference
            cfg=Experiment();r=run_reference(cfg,args.output,args.particles,20,args.seed,reuse=False)
            print(json.dumps({k:v for k,v in r.items() if k not in {"mean","stderr"}},indent=2));return
        if args.command=="predict":
            from .model import FrozenSurrogate
            cfg=Experiment(**json.loads(args.config.read_text())) if args.config else Experiment()
            model=FrozenSurrogate(project_root()/"data/model.npz");pred,half=model.predict(cfg.vector())
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,prediction=pred.reshape(2,20,24),halfwidth_90=half.reshape(2,20,24))
            print(json.dumps({"kind":"new_model_prediction","output":str(args.output),"ood":bool(model.out_of_domain(cfg.vector(),cfg.family)[0]),"units":"cm^-2 per source particle"}));return
        if args.command=="generate":
            from .dataset import generate
            return generate(args.output,args.cache)
        if args.command=="train":
            from .study import study
            study(args.project);return
        if args.command=="validate":
            from .validation import validate
            validate(args.output,args.cache);return
    except (RuntimeError,ValueError,TypeError,OSError) as exc:
        parser.exit(2,f"fluxatlas: {exc}\n")

if __name__=="__main__":main()

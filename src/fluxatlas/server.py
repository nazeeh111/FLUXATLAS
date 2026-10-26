"""Loopback-only observatory server. Static replay works without this server."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import shutil
import threading
import time
from urllib.parse import urlparse
from .config import Experiment
from .model import FrozenSurrogate
from .transport import run_reference


def serve(project,port=8765,cache=None):
    project=Path(project).resolve()
    model_path=project/"data/model.npz"
    model=FrozenSurrogate(model_path) if model_path.exists() else None
    cache=Path(cache or Path.cwd()/".cache/fluxatlas")
    solver_lock=threading.Lock()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,directory=str(project/"web"),**kwargs)
        def respond(self,status,value):
            content=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(content)));self.send_header("Cache-Control","no-store")
            self.end_headers();self.wfile.write(content)
        def valid_host(self):
            return self.headers.get("Host","").split(":")[0] in {"127.0.0.1","localhost"}
        def do_GET(self):
            if not self.valid_host():return self.respond(403,{"error":"Loopback Host header required"})
            if self.path=="/api/status":
                try:import openmc;available=bool(shutil.which("openmc"))
                except ImportError:available=False
                return self.respond(200,{"prediction":model is not None,"solver":available,"mode":"local","interval_level":.9})
            if self.path=="/technical-report.md":
                content=(project/"docs/technical-report.md").read_bytes()
                self.send_response(200);self.send_header("Content-Type","text/markdown; charset=utf-8");self.send_header("Content-Length",str(len(content)));self.end_headers();self.wfile.write(content);return
            return super().do_GET()
        def do_POST(self):
            if not self.valid_host():return self.respond(403,{"error":"Loopback Host header required"})
            origin=self.headers.get("Origin")
            if origin and urlparse(origin).netloc!=self.headers.get("Host"):
                return self.respond(403,{"error":"Same-origin requests required"})
            if self.path not in {"/api/predict","/api/simulate"}:return self.respond(404,{"error":"Unknown API endpoint"})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=16384:raise ValueError("JSON request must be 1..16384 bytes")
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError("JSON object required")
                config=Experiment(**payload.get("config",payload))
                if self.path=="/api/predict":
                    if model is None:return self.respond(503,{"error":"Trained model artifact unavailable"})
                    start=time.perf_counter();pred,half=model.predict(config.vector())
                    return self.respond(200,{"prediction":pred.reshape(2,-1).tolist(),"uncertainty":half.reshape(2,-1).tolist(),
                        "ood":bool(model.out_of_domain(config.vector(),config.family)[0]),"seconds":time.perf_counter()-start,
                        "config":config.to_dict(),"kind":"new_model_prediction","interval_level":.9})
                particles=payload.get("particles",2000);seed=payload.get("seed",123457)
                if isinstance(particles,bool) or not isinstance(particles,int) or not 100<=particles<=5000:
                    raise ValueError("Local API supports 100..5000 particles per batch, 20 batches")
                if not solver_lock.acquire(blocking=False):return self.respond(409,{"error":"A solver run is already in progress"})
                try:result=run_reference(config,cache,particles,20,seed,reuse=False)
                finally:solver_lock.release()
                return self.respond(200,{"reference":result["mean"].reshape(2,-1).tolist(),"stderr":result["stderr"].reshape(2,-1).tolist(),
                    "seconds":result["seconds"],"config":config.to_dict(),"seed":seed,"histories":result["histories"],"kind":"new_openmc_execution"})
            except (ValueError,TypeError,json.JSONDecodeError) as exc:return self.respond(400,{"error":str(exc)})
            except RuntimeError as exc:return self.respond(503,{"error":str(exc)})
            except Exception as exc:
                self.log_error("Solver/API error: %s",exc)
                return self.respond(500,{"error":"Computation failed; inspect local server log"})
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"FLUXATLAS local observatory: http://127.0.0.1:{server.server_port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

"""Reopen this saved example with NumPy; no model training or transport execution."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
for name,expected in json.loads((P/'MANIFEST.json').read_text()).items():
    if hashlib.sha256((P/name).read_bytes()).hexdigest()!=expected:
        raise ValueError('Changed example member: '+name)
with np.load(P/'prediction.npz',allow_pickle=False) as a:
    predicted=a['prediction'];halfwidth=a['halfwidth_90']
with np.load(P/'reference/field.npz',allow_pickle=False) as a:
    reference=a['mean'];stderr=a['stderr']
if predicted.shape!=reference.shape or reference.shape!=(2,20,24):
    raise ValueError('Unexpected fixed grid')
error=predicted-reference
print(json.dumps({'relative_l2':float(np.linalg.norm(error)/np.linalg.norm(reference)),
 'slow_relative_l2':float(np.linalg.norm(error[0])/np.linalg.norm(reference[0])),
 'fast_relative_l2':float(np.linalg.norm(error[1])/np.linalg.norm(reference[1])),
 'whole_field_interval_covered':bool((abs(error)<=halfwidth).all()),
 'scope':'Retained finite-history simulation estimate; no new OpenMC, calibration or physical measurement'},indent=2))

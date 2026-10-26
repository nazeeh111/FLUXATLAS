#!/usr/bin/env python3
"""Prepare the deterministic film after archive and screenshot are final."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('screenshot',type=Path);args=p.parse_args()
if not args.screenshot.is_file():p.error('A genuine current-source observatory screenshot is required')
subprocess.run([sys.executable,str(root/'export-evidence.py')],check=True)
shutil.copyfile(args.screenshot,root/'composition/assets/observatory.png')
for i in range(1,5):shutil.copyfile(root.parent/'web/mark.svg',root/f'composition/assets/mark-{i}.svg')
shutil.copyfile(root/'composition/assets/reference.svg',root/'composition/assets/reference-hook.svg')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(args.screenshot),'-vf','crop=1470:1120:0:0','-frames:v','1',str(root/'composition/assets/observatory-crop.png')],check=True)
meta=json.loads((root/'composition/assets/provenance.json').read_text())
meta['screenshot_sha256']=hashlib.sha256(args.screenshot.read_bytes()).hexdigest()
meta['screenshot_crop_pixels']={'x':0,'y':0,'width':1470,'height':1120}
(root/'composition/assets/provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
s=(root/'composition/index.html.template').read_text()
for a,b in {'##CASE##':meta['case_id'],'##HISTORIES##':f"{meta['histories']:,}",'##ERRORMAX##':f"{meta['error_upper']:.2e}",'##REPEATS##':str(meta['trial_repeats'])}.items():s=s.replace(a,b)
(root/'composition/index.html').write_text(s)
print(root/'composition/index.html')

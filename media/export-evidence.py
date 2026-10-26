#!/usr/bin/env python3
"""Export launch visuals strictly from the completed numerical archive.

No fitting, interpolation, smoothing or invented measurements is done here.
Each cell is colored directly from the stored field. The selected case is
nearest the median held-out relative L2, not the best-looking outcome.
"""
from pathlib import Path
import hashlib
import json
import math
from statistics import mean, median

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(__file__).resolve().parent/'composition/assets'
DEST.mkdir(exist_ok=True,parents=True)
archive=ROOT/'web/atlas.json'
atlas=json.loads(archive.read_text())
test=[c for c in atlas['cases'] if c['split']=='test']
mid=median(c['metrics']['relative_l2'] for c in test)
case=min(test,key=lambda c:abs(c['metrics']['relative_l2']-mid))
nx,ny=atlas['grid']['nx'],atlas['grid']['ny']

def flat(field):
    if len(field)==2:return [a+b for a,b in zip(field[0],field[1])]
    n=nx*ny
    return [a+b for a,b in zip(field[:n],field[n:])]
ref,pred=flat(case['reference']),flat(case['prediction'])
error=[abs(a-b) for a,b in zip(pred,ref)]
colors=['#102b43','#245c76','#5c9b9b','#c1d9b1','#f0cf79','#d66339']

def color(v,lo,hi,log=False):
    if log:
        f=(math.log10(max(v,lo))-math.log10(lo))/(math.log10(hi)-math.log10(lo))
    else:f=(v-lo)/(hi-lo or 1)
    q=min(max(f,0),1)*(len(colors)-1);i=min(int(q),len(colors)-2);t=q-i
    a,b=colors[i],colors[i+1]
    rgb=[round(int(a[k:k+2],16)*(1-t)+int(b[k:k+2],16)*t) for k in (1,3,5)]
    return '#'+''.join(f'{v:02x}' for v in rgb)

hi=max(max(ref),max(pred));lo=max(hi*1e-4,1e-12)
def heatmap(values,name,upper,log):
    left,top,w,h=55,22,720,420
    bits=[f'<svg xmlns="http://www.w3.org/2000/svg" width="840" height="510" viewBox="0 0 840 510"><rect width="840" height="510" fill="#faf9f4"/>']
    for y in range(ny):
        for x in range(nx):
            v=values[y*nx+x]
            bits.append(f'<rect x="{left+x*w/nx:.3f}" y="{top+(ny-1-y)*h/ny:.3f}" width="{w/nx+.1:.3f}" height="{h/ny+.1:.3f}" fill="{color(v,lo if log else 0,upper,log)}"/>')
    cfg=case['config'];sx=left+6/20*w;sy=top+(8-6)/16*h;sw=cfg['thickness']/20*w;sh=12/16*h
    bits.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" stroke="#f4f2e9" stroke-dasharray="7 5" stroke-width="2" fill="none"/>')
    bits.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" stroke="#172d36" fill="none"/>')
    for x in (0,5,10,15,20):bits.append(f'<text x="{left+x/20*w}" y="{top+h+29}" text-anchor="middle" fill="#172d36" font-family="monospace" font-size="17">{x}</text>')
    for y in (-8,-4,0,4,8):bits.append(f'<text x="{left-12}" y="{top+(8-y)/16*h+6}" text-anchor="end" fill="#172d36" font-family="monospace" font-size="17">{y}</text>')
    bits.append(f'<text x="{left+w/2}" y="{top+h+56}" text-anchor="middle" fill="#172d36" font-family="monospace" font-size="17">x / cm</text>')
    bits.append(f'<text transform="translate(17 232) rotate(-90)" text-anchor="middle" fill="#172d36" font-family="monospace" font-size="17">y / cm</text>')
    bits.append('</svg>')
    (DEST/name).write_text(''.join(bits))

heatmap(ref,'reference.svg',hi,True);heatmap(pred,'prediction.svg',hi,True);heatmap(error,'error.svg',max(error),False)
# One shared flux scale, with exact data limits.
(DEST/'legend.svg').write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1640" height="60"><defs><linearGradient id="r">{''.join(f'<stop offset="{i/(len(colors)-1)}" stop-color="{c}"/>' for i,c in enumerate(colors))}</linearGradient></defs><rect x="250" y="9" width="850" height="12" fill="url(#r)"/><text x="230" y="23" text-anchor="end" font-size="19" font-family="monospace" fill="#172d36">{lo:.2e}</text><text x="1120" y="23" font-size="19" font-family="monospace" fill="#172d36">{hi:.2e}</text><text x="820" y="53" text-anchor="middle" font-size="17" font-family="monospace" fill="#5d686d">Shared log scale · cm⁻² per source particle</text></svg>''')
# History-matched acquisition curves from actual study rows, without smoothing.
rows=atlas['benchmark']['active_learning']
groups={s:{} for s in ('active','random')}
for row in rows:groups[row['strategy']].setdefault(row['budget_histories'],[]).append(row['relative_l2'])
xs=sorted({r['budget_histories'] for r in rows});ys=[mean(v) for vals in groups.values() for v in vals.values()]
ymin=max(0,min(ys)*.8);ymax=max(ys)*1.15
W,H=940,500;l,t,w,h=100,50,770,350
fx=lambda x:l+(x-min(xs))/(max(xs)-min(xs))*w
fy=lambda y:t+h-(y-ymin)/(ymax-ymin)*h
b=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="{W}" height="{H}" fill="#faf9f4"/>']
for i in range(5):
    v=ymin+i/4*(ymax-ymin);yy=fy(v)
    b.append(f'<path d="M{l},{yy}H{l+w}" stroke="#d9ded9"/><text x="{l-15}" y="{yy+7}" fill="#172d36" text-anchor="end" font-family="monospace" font-size="19">{100*v:.1f}%</text>')
for x in xs:b.append(f'<text x="{fx(x)}" y="{t+h+34}" text-anchor="middle" fill="#172d36" font-family="monospace" font-size="18">{x/1e6:.2f}M</text>')
for s,c in [('active','#d86435'),('random','#3a7383')]:
    points=[(fx(x),fy(mean(groups[s][x]))) for x in xs]
    b.append(f'<polyline points="'+ ' '.join(f'{a},{bb}' for a,bb in points)+f'" fill="none" stroke="{c}" stroke-width="4"'+(' stroke-dasharray="10 7"' if s=='random' else '')+'/>')
    for a,bb in points:b.append(f'<circle cx="{a}" cy="{bb}" r="5" fill="{c}"/>')
b.append(f'<text x="{l+w/2}" y="{H-42}" text-anchor="middle" fill="#172d36" font-family="monospace" font-size="18">Particle histories, including the low-fidelity pool</text>')
b.append(f'<text x="{l}" y="26" fill="#d86435" font-size="18" font-family="sans-serif">Active selection</text><text x="{l+215}" y="26" fill="#3a7383" font-size="18" font-family="sans-serif">Random selection</text>')
b.append('</svg>');(DEST/'acquisition.svg').write_text(''.join(b))
meta={'protocol_version':atlas['benchmark'].get('protocol_version'),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'case_id':case['id'],'selection_rule':'Held-out case nearest the median both-group field relative L2','configuration':case['config'],'histories':case['histories'],'seed':case['seed'],'case_relative_l2':case['metrics']['relative_l2'],'reference_flux_limits':[lo,hi],'error_upper':max(error),'trial_repeats':len({r['repeat'] for r in rows}),'source':'web/atlas.json','normalization':atlas['units'],'no_data_smoothing':True}
(DEST/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))

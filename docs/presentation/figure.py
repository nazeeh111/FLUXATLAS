"""Render every cell of one retained comparison. No solver, fitting or smoothing.
Reference/prediction share one positive logarithmic scale; signed error shares
one zero-centered linear scale across both energy groups. Zero is explicit gray.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm,Normalize,LinearSegmentedColormap
P=Path(__file__).resolve().parent;c=json.loads((P/'case.json').read_text());checks=json.loads((P/'checks.json').read_text());ref=np.array(c['reference']).reshape(2,20,24);pred=np.load(P/'prediction.npz')['prediction'];pred=pred.reshape(2,20,24);delta=pred-ref;x=np.linspace(0,20,25);y=np.linspace(-8,8,21);positive=np.concatenate([ref[ref>0],pred[pred>0]]);lo=float(positive.min());hi=float(positive.max());bound=float(abs(delta).max());field=LinearSegmentedColormap.from_list('atlas',['#102b43','#245c76','#5c9b9b','#c1d9b1','#f0cf79','#d66339']).with_extremes(bad='#d3d7d4');signed=LinearSegmentedColormap.from_list('signed',['#24567c','#f4f3ed','#bc4f2f']);rows=[]
with mpl.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'svg.fonttype':'none','svg.hashsalt':'fluxatlas-test120','path.simplify':False,'axes.spines.top':False,'axes.spines.right':False}):
 for phone in (False,True):
  w,h=(460,2630) if phone else (1240,1120);fig=plt.figure(figsize=(w/100,h/100),dpi=100,facecolor='#f6f5ee');base=fig.add_axes([0,0,1,1]);base.axis('off')
  def text(xp,yp,s,size=17,weight='normal',color='#18343d'):
   fig.text(xp/w,1-yp/h,s,fontsize=size*.72,fontweight=weight,color=color,va='center')
  if phone:
   text(24,36,'FLUXATLAS / RETAINED FIELD, FRESH QUERY',12,'bold');text(24,91,'5.56% overall.',35,'bold');text(24,135,'21.66% in slow.',35,'bold');text(24,180,'One aggregate can hide an energy-group error.',15);text(24,215,'test-120 · 40,000 retained OpenMC histories',14)
  else:
   text(40,40,'FLUXATLAS / RETAINED FIELD, FRESH QUERY',14,'bold');text(40,98,'One number hides the group that needs work.',38,'bold');text(40,151,'5.56% combined error   ·   21.66% slow   ·   3.86% fast',26,'bold');text(40,191,'test-120 · retained OpenMC reference + current frozen weights · synthetic materials',17)
  for gi,group in enumerate(['Slow','Fast']):
   for col,(label,data) in enumerate([('OpenMC reference',ref[gi]),('Fresh model prediction',pred[gi]),('Signed model − reference',delta[gi])]):
    if phone:xx,yy,ww,hh=72,290+(gi*3+col)*320+gi*60,350,230
    else:xx,yy,ww,hh=70+col*410,295+gi*350,335,238
    if col==0:text(24 if phone else 40,yy-57,f'{group}: {"1e-5 eV–1 keV" if gi==0 else "1 keV–20 MeV"}',15,'bold')
    text(xx,yy-25,label,15,'bold')
    ax=fig.add_axes([xx/w,(h-yy-hh)/h,ww/w,hh/h]);arr=np.ma.masked_less_equal(data,0) if col<2 else data;im=ax.pcolormesh(x,y,arr,shading='flat',cmap=field if col<2 else signed,norm=LogNorm(lo,hi) if col<2 else Normalize(-bound,bound));im.set_gid(f'{group.lower()}-{col}-cells');ax.set(xlim=(0,20),ylim=(-8,8),xticks=[0,10,20],yticks=[-8,0,8]);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x / cm',fontsize=10);ax.set_ylabel('y / cm',fontsize=10);ax.tick_params(labelsize=10);ax.set_facecolor('#d3d7d4')
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((6,-6),c['config']['thickness'],12,fill=False,edgecolor='#ffffff',linestyle='--',linewidth=.8));rows.append({'figure':'phone' if phone else 'desktop','group':group,'panel':label,'cells':480})
  if phone:
   cf=fig.add_axes([.15,.105,.72,.009]);ce=fig.add_axes([.15,.065,.72,.009]);fig.colorbar(mpl.cm.ScalarMappable(norm=LogNorm(lo,hi),cmap=field),cax=cf,orientation='horizontal',format=FuncFormatter(lambda v,pos:f'{v:.0e}')).set_label('Reference/model: shared log flux scale',fontsize=10);fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-bound,bound),cmap=signed),cax=ce,orientation='horizontal',ticks=[-bound,0,bound]).set_label('Signed error: shared linear scale',fontsize=10);text(24,2562,'cm^-2 per source particle · zero is gray',14);text(24,2597,'Simulation estimates, not dose or physical truth.',13)
  else:
   cf=fig.add_axes([.07,.115,.46,.015]);ce=fig.add_axes([.67,.115,.29,.015]);fig.colorbar(mpl.cm.ScalarMappable(norm=LogNorm(lo,hi),cmap=field),cax=cf,orientation='horizontal',format=FuncFormatter(lambda v,pos:f'{v:.0e}')).set_label('Reference/model: one shared positive log scale',fontsize=11);fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-bound,bound),cmap=signed),cax=ce,orientation='horizontal',ticks=[-bound,0,bound]).set_label('Signed error: one shared linear scale',fontsize=11);text(40,1044,'All 2 × 480 cells retained · cm^-2 per source particle · zero predictions are gray · dashed outline: declared synthetic shield',15);text(40,1090,'Every cell is inside a wide model interval here (mean half-width 1.88 × mean flux). That is not a new calibration result.',15)
  name='group-error-mobile' if phone else 'group-error';fig.savefig(P/(name+'.png'),dpi=120);fig.savefig(P/(name+'.svg'),metadata={'Date':None});plt.close(fig)
  svg=(P/(name+'.svg')).read_text();import re
  svg=re.sub(r'font-family:[^;\"]+',"font-family: 'DejaVu Sans', Arial, sans-serif",svg);i=svg.index('>',svg.index('<svg '))+1;svg=svg[:i]+'<style>text,tspan{font-family:Arial,Helvetica,sans-serif!important}</style><title>One aggregate hides the slow-group error</title><desc>All 480 cells per energy group, retained OpenMC reference and fresh frozen-model prediction, plus signed error. Combined relative L2 is 5.56 percent, slow 21.66 percent, fast 3.86 percent. Shared field and error scales; two zero predictions gray. No new transport run or physical accuracy claim.</desc>'+svg[i:];(P/(name+'.svg')).write_text(svg)
manifest={'source_commit':checks['source_commit'],'case':'test-120','cells_per_data_panel':480,'six_panels_per_figure':True,'field_scale':{'kind':'shared LogNorm across all four reference/model panels','vmin':lo,'vmax':hi,'zeros':'gray'},'signed_error_scale':{'kind':'shared linear symmetric','vmin':-bound,'vmax':bound},'raw_source_hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['case.json','prediction.npz','checks.json','figure.py']},'exports':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['group-error.png','group-error.svg','group-error-mobile.png','group-error-mobile.svg']}}
(P/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Rendered both six-panel figures; all480 cells per panel')

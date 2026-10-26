"""Exact mesh-integrated uncollided beam for the supported solid-shield family.

This calculates only particles with no collision, not the total scattering
solution. Its deterministic physics component is added to a learned scattered
field. Source is uniform over y +/-0.4 cm, monodirectional +x at x=0.5 cm.
"""
import numpy as np
from .config import X_EDGES,Y_EDGES,NX,NY,BOUNDS


def ballistic_field(x):
    x=np.atleast_2d(x)
    physical=x*np.ptp(BOUNDS,axis=1)+BOUNDS[:,0]
    result=np.zeros((len(x),2,NY,NX))
    for n,(thickness,density,source_y,fast_fraction) in enumerate(physical):
        # Per-cell fraction of the source beam, divided by transverse cell area.
        overlap=np.maximum(0,np.minimum(Y_EDGES[1:],source_y+.4)-np.maximum(Y_EDGES[:-1],source_y-.4))/.8
        transverse=overlap/np.diff(Y_EDGES)
        for g,(background,shield,weight) in enumerate(((.04,1.1*density,1-fast_fraction),(.025,.65*density,fast_fraction))):
            for ix,(left,right) in enumerate(zip(X_EDGES[:-1],X_EDGES[1:])):
                if right<=.5:continue
                cuts=sorted(set([max(left,.5),right]+[p for p in (6.,6.+thickness) if max(left,.5)<p<right]))
                integral=0.
                for a,b in zip(cuts[:-1],cuts[1:]):
                    shield_length=max(0.,min(a,6.+thickness)-6.)
                    bg_length=a-.5-shield_length
                    entering=np.exp(-background*bg_length-shield*shield_length)
                    sigma=shield if 6.<(a+b)/2<6.+thickness else background
                    integral+=entering*(-np.expm1(-sigma*(b-a)))/sigma
                result[n,g,:,ix]=weight*transverse*integral/(right-left)
    return result.reshape(len(x),-1)

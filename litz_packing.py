"""
litz_packing module 

Contains helper functions and functions to create different litz wire packings
and cross sections using different packing strategies.
"""

import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import Voronoi 

def cocentric_packing(nstrands, strand_diam, strand_ins, out_ins,
                      **kwargs):
    """ Cocentric packing, not as space efficient as hexagonal but 
    can be used to create litz wires with exactly the wanted number of strands"""

    #strand_diam = 0.75e-3 # strand diameter
    srad = strand_diam/2 
    
    # approximate how many layers and how many strands per layer there will be    

    layers = []
    scum = 0
    
    # sum of angles of regular polygon -> (n-2)*pi
    # one angle -> (n-2)*pi/n
    # calculate the radius of the layer where there are n optimally packed
    # strands, i.e. the starting radius (can be used to set/unset the middle
    # strand) 
    n = kwargs.get('first_layer', 1)
    
    # set the first layer radiu
    if n > 1:
        rad0 = srad*(1+strand_ins)/np.cos((n-2)*np.pi/(2*n))
    else:
        rad0 = 0

    layers.append((n, rad0))

    i=1
    # scum is the accumulated number of strands
    # this adds strands to each layer as long as needed
    while scum < nstrands:
        rad = rad0 + i*(2*(srad*(1+strand_ins)))
        
        # solve n from equation of rad0
        acos = np.arccos(srad*(1+strand_ins)/rad)
        nstr = int(2*np.pi/(np.pi-2*acos))

        layers.append((nstr, rad))
        scum = scum + nstr
        i = i + 1

    max_strands = np.sum([l for (l, _) in layers])
    nlayers = len(layers)    
    
    strategy = kwargs.get('strategy', None)
    if strategy == None:
        raise RuntimeError("Pleace specify 'strategy' keyword argument!")
    if 'fill_evenly' in strategy:
        fill_factor = nstrands/max_strands
    elif 'fill_inside_out' in strategy:
        fill_factor = 1
    else:
        raise RuntimeError(f"Unknown filling strategy {strategy}!" 
                            "Options: fill_evenly/fill_inside_out")

    # possibility to filter layers to only contain odd/even number of 
    # strands
    if 'even_layers' in strategy:
        nstrand_filter = lambda x: x if x%2==0 else x-1
    elif 'odd_layers' in strategy: 
        nstrand_filter = lambda x: x if x%2==1 else x-1
    else:
        nstrand_filter = lambda x: x
    
    if fill_factor > 1:
        raise RuntimeError(f"Can't fit {nstrands} into bundle. "
                           f"Maximum is {max_strands}")
        
    # fill each layer approximately up to fill_factor, from in to out 
    # fill first layer up to n though
        
    strands = []
    for nstr, rad in layers[0:-1]:
        # max fills the first layer up to n 
        nstr = int(max(n, nstrand_filter(int(fill_factor*nstr))))
        
        angles = np.arange(0,nstr)*2*np.pi/nstr
        for a in angles:
            x = np.cos(a)*rad
            y = np.sin(a)*rad
            strands.append((x,y))
    
    strands_thus_far = len(strands)
    _, rad = layers[-1]
    nstr = nstrands - strands_thus_far
    
    angles = np.arange(0,nstr)*2*np.pi/nstr
    for a in angles:
        x = np.cos(a)*rad
        y = np.sin(a)*rad
        strands.append((x,y))
    
    maxd = np.max(np.linalg.norm(strands, axis=1))
    wire_diameter = 2*(maxd + (srad*(1+strand_ins)))*(1+out_ins)
    
    return wire_diameter, strand_diam, np.array(strands)

           
def hexagonal_packing(nseeds, strand_diam, sinsu, winsu, **kwargs):
    """ Make a hexagonal packing of litz strands.
    Inputs:
        nseeds - Int, defines the numer of points in the initial meshgrid,
                 use to control the amount of strands
        strand_diam - strand outer diameter,
        sinsu - strand insulation thickness in % of the strand diameter
        winsu - outer insulation thickness in % of the wire diameter
    """

    seeds = np.linspace(-1,1,nseeds)
    dx = seeds[1]-seeds[0]
        
    xs, ys = np.meshgrid(seeds, seeds)

    if (nseeds-1) % 4 == 0:
        ys[:,1::2] = ys[:,1::2] + 0.5*dx;
    else:
        ys[:,0::2] = ys[:,0::2] + 0.5*dx;
    
    ys = ys*2/np.sqrt(3);
    
    points = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    
    vor = Voronoi(points)

    hexs = [v for v in vor.regions if len(v) == 6]
    all_cells = vor.vertices[hexs, :]
    max_dists = np.max(np.linalg.norm(all_cells, axis=2), axis=1)
    cells = all_cells[max_dists < 1, :]

    
    strand_cps = np.mean(cells, axis=1)
    
    # if strand bundle is not symmetric, it will be off center so...
    # move it back to center
    strand_cps = strand_cps - np.mean(strand_cps, axis=0)
    
    # quite a silly way to calculate the strand diameter.but it indeed is 
    # the minimum of the distances from the first cell center to all the rest
    strand_diam_unsc = np.min(np.linalg.norm(strand_cps[1:]-strand_cps[0], 
                                             axis=1))
#    nstrands = len(strand_cps)
    
    # scaling factor to scale the geometry to the strand size we want
    scale = (strand_diam*(1+sinsu))/strand_diam_unsc
    
    strand_cps_scaled = strand_cps*scale
    strand_diameter_scaled = strand_diam
    
    # compute wide diameter
    maxd = np.max(np.linalg.norm(strand_cps_scaled, axis=1))
    wire_diameter_scaled = 2*(maxd + strand_diameter_scaled/2*(1+sinsu))*(1+winsu)

    
    return wire_diameter_scaled, strand_diameter_scaled, strand_cps_scaled

def hexagonal_packing_cross_section(nseeds, Areq, insu, out_insu):
    """ Make a hexagonal packing and scale the result to be Areq cross section
        
    Parameter insu must be a percentage of the strand radius. 
    out_insu is the insulation thickness around the wire as meters
    Returns:
        (wire diameter, strand diameter, strand center points)
    """
    
    seeds = np.linspace(-0.5, 0.5,nseeds)
    dx = seeds[1]-seeds[0]
        
    xs, ys = np.meshgrid(seeds, seeds)

    if (nseeds-1) % 4 == 0:
        ys[:,1::2] = ys[:,1::2] + 0.5*dx;
    else:
        ys[:,0::2] = ys[:,0::2] + 0.5*dx;
    
    ys = ys*2/np.sqrt(3);
    
    points = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    
    vor = Voronoi(points)

    hexs = [v for v in vor.regions if len(v) == 6]
    all_cells = vor.vertices[hexs, :]
    max_dists = np.max(np.linalg.norm(all_cells, axis=2), axis=1)
    cells = all_cells[max_dists < 0.5, :]
 
    strand_cps = np.mean(cells, axis=1)
    
    # if strand bundle is not symmetric, it will be off center so...
    # move it back to center
    strand_cps = strand_cps - np.mean(strand_cps, axis=0)
    
    # quite a silly way to calculate the strand diameter.but it indeed is 
    # the minimum of the distances from the first cell center to all the rest
    # minus the insulation thickness
    strand_diam = np.min(np.linalg.norm(strand_cps[1:]-strand_cps[0], axis=1))*(1-insu)
    nstrands = len(strand_cps)
    
    Acu = nstrands*(strand_diam/2)**2*np.pi
    scale = np.sqrt(Areq/Acu)
    
    strand_cps_scaled = scale*strand_cps
    strand_diam_scaled = scale*strand_diam
    
    wire_diameter = (np.max(np.linalg.norm(strand_cps_scaled, axis=1), axis=0)*2 
                     + strand_diam_scaled*(1+insu)/(1-insu)
                     + out_insu)
    
    return wire_diameter, strand_diam_scaled, strand_cps_scaled


def draw_litz_wire(ax, wire_diameter, strand_diameter, strands, origin=(0,0)):
    """
    Draw the litz wire cross section to axis ax
    """
    nstrands = len(strands)
    o = np.array(origin)

    for cp in strands:
        c = plt.Circle(cp+o, strand_diameter/2, fill=False, color='r')
        ax.add_artist(c)
        
    c = plt.Circle(o, wire_diameter/2, fill=False, color='b')
    ax.add_artist(c)
    
    wd = wire_diameter/2
    xlims = wd*np.array([-1,1])+o[0]
    ylims = wd*np.array([-1,1])+o[1]
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect('equal')
    

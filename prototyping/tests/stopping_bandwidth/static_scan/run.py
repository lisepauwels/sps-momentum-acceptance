from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xobjects as xo
import xpart as xp
import xcoll as xc
import json
import sys

Qx = float(sys.argv[1])
Qy = float(sys.argv[2])

# Line configuration
line = xt.load(Path('/Users/lisepauwels/phd/code/sps-xsuite-model/sps_with_aperture_inj_q20_beam_sagitta4.json'))
env = line.env
cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
for name in cavity_names:
    line[name].frequency = 200e6
    line[name].lag = 180
    line[name].voltage = 0
# line['acl.31735'].voltage = 0 #setting 800 cav to 0V
line['actcse.31632'].voltage = 3.0e6

qx = Qx
qy = Qy
xi_x = 0.5
xi_y = 0.5

line.match(
        method="6d",
        vary=[
            xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad"),
            xt.VaryList(["qph_setvalue", "qpv_setvalue"], step=1e-4, tag="sext"),
        ],
        targets=[
            xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune"),
            xt.TargetSet(dqx=xi_x * qx, dqy=xi_y * qy, tol=1e-2, tag="chrom"),
        ],
    )
tw = line.twiss()

num_turns = 6000
num_particles = 500
nemitt_x = 2e-6
nemitt_y = 2e-6
sigma_z = 0.224

#Generating the particles
part = xp.generate_matched_gaussian_bunch(nemitt_x=nemitt_x,
                                        nemitt_y=nemitt_y,
                                        sigma_z=sigma_z, num_particles=num_particles, line=line)

#Tracking
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
line.scattering.enable()
line.track(particles=part, num_turns=num_turns, time=True, with_progress=5)
line.scattering.disable()

savings = {}

savings['delta'] = part.delta.copy()
savings['state'] = part.state.copy()
savings['at_turn'] = part.at_turn.copy()

with open(f'outputs/stopping_bandwidth_Qx{Qx}_Qy{Qy}.json', 'w') as f:
    json.dump(savings, f, cls=xo.JEncoder)
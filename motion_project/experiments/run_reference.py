"""Reference CondMDI run per clip: full body of the original every 5 frames (the repo's benchmark_sparse T=5).
Not an HPP experiment - it measures the model's own baseline behaviour (e.g. jitter) for acceptance C7.

python -m motion_project.experiments.run_reference
"""
import numpy as np
from motion_project.experiments.common import load_cfg, run_reference
from motion_project.realizer.condmdi_adapter import CondMDIRealizer

if __name__ == '__main__':
    ccfg = load_cfg('condmdi')
    realizer = CondMDIRealizer(ccfg)
    for c in load_cfg('experiment')['clips']:
        r = run_reference(realizer, c['id'], ccfg)
        print(c['id'], 'jerk', np.round(r['jerk_mps3'], 1), 'constraint err mean (cm)',
              [round(100 * x['mean'], 1) for x in r['constraint_vs_original']])

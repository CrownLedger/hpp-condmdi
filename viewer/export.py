"""Export every results.npy under save/results to viewer/data.json for the three.js viewer."""
import glob, json, os, numpy as np


def frames_of(motion, T):
    # (22, 3, T) -> T x 22 x 3 nested lists
    return np.round(motion[:, :, :T].transpose(2, 0, 1), 4).tolist()


runs = []
for f in sorted(glob.glob('save/results/**/results.npy', recursive=True)):
    d = np.load(f, allow_pickle=True).item()
    m, L = d['motion'], d['lengths']          # (reps, n, 22, 3, T), (reps, n)
    n = m.shape[1]
    for s in range(n):
        T = int(L[0, s])
        run = {'run': os.path.basename(os.path.dirname(f)) + (f' [sample {s}]' if n > 1 else ''),
               'text': str(d['text'][s]), 'input': None,
               'outputs': [{'rep': r, 'frames': frames_of(m[r, s], int(L[r, s]))} for r in range(m.shape[0])]}
        mask = d.get('observed_mask')
        # Only runs that were actually conditioned on keyframes have a meaningful input motion.
        if mask is not None and mask[s].any():
            obs = mask[s, :, 0, :T].astype(bool)  # (22, T)
            run['input'] = {'frames': frames_of(d['observed_motion'][s], T),
                            'keyjoints': [np.nonzero(obs[:, t])[0].tolist() for t in range(T)]}
        runs.append(run)
json.dump({'fps': 20, 'runs': runs}, open('viewer/data.json', 'w'))
print(len(runs), 'runs exported:', [(r['run'], r['input'] is not None) for r in runs])

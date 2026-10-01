"""Convert tabular diagonal deformations to independent nominal-stress rows."""
import numpy as np
from .types import Observations


def isotropic_observations(frame, *, components=('P11', 'P22')):
    Fs, targets, stress_components, modes, groups = [], [], [], [], []
    for index, row in frame.iterrows():
        stretches = row[['lambda1', 'lambda2', 'lambda3']].to_numpy(float)
        if np.any(stretches <= 0) or not np.all(np.isfinite(stretches)):
            raise ValueError(f"Invalid stretches in row {index}")
        for name, component in (('P11', (0, 0)), ('P22', (1, 1))):
            if name not in components or not np.isfinite(row[name]):
                continue
            # EBT P22 copied from P11 is a symmetry prediction, not a second
            # independent measurement. UT transverse zero is a boundary condition.
            if name == 'P22' and row['dataset'] != 'Kawabata':
                continue
            Fs.append(np.diag(stretches))
            targets.append(float(row[name]))
            stress_components.append(component)
            modes.append(f"{row['dataset']}:{row['mode']}")
            groups.append(f"{row['dataset']}:{row['mode']}:{index}")
    return Observations(np.asarray(Fs), np.asarray(targets), tuple(stress_components),
                        np.asarray(modes), np.asarray(groups), 'nominal', 'MPa')

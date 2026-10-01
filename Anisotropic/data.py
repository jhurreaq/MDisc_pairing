"""Sommer/Martonová cardiac data: Cauchy stresses in kPa.

The workbook's generic 'P' headers are retained by the upstream dataset, whose
publication defines these measurements as Cauchy stresses. See docs/MECHANICS.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from mdisc.types import Observations

SHEAR_COLUMNS = {'fs': (0, 1, (1, 0)), 'fn': (2, 3, (2, 0)),
                 'sf': (5, 6, (0, 1)), 'sn': (7, 8, (2, 1)),
                 'nf': (10, 11, (0, 2)), 'ns': (12, 13, (1, 2))}
BIAXIAL_COLUMNS = {'1:1': (15, 16, 17, 18), '1:0.75': (20, 21, 22, 23),
                   '0.75:1': (25, 26, 27, 28), '1:0.5': (30, 31, 32, 33),
                   '0.5:1': (35, 36, 37, 38)}


def load_data(input_dir, *, sheet='Sheet1'):
    path = Path(input_dir) / 'CANNsHEARTdata_shear05.xlsx'
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    Fs, targets, components, modes, groups = [], [], [], [], []
    records = []
    for mode, (gcol, stresscol, shear) in SHEAR_COLUMNS.items():
        for row in range(4, len(df)):
            values = pd.to_numeric(df.iloc[row, [gcol, stresscol]], errors='coerce').to_numpy(float)
            if np.isnan(values).all():
                continue
            if not np.isfinite(values).all():
                raise ValueError(f'{path}: incomplete shear measurement at row {row+1}, mode {mode}')
            gamma, stress = values
            F = np.eye(3)
            F[shear] = gamma
            component = tuple(sorted(shear))  # Cauchy stress is symmetric
            Fs.append(F)
            targets.append(stress)
            components.append(component)
            modes.append(mode)
            groups.append(f'{mode}:{row}')
            records.append({'mode': mode, 'x': gamma, 'target': stress, 'row': row})
    for ratio, cols in BIAXIAL_COLUMNS.items():
        for row in range(4, len(df)):
            values = pd.to_numeric(df.iloc[row, list(cols)], errors='coerce').to_numpy(float)
            if np.isnan(values).all():
                continue
            lf, sf, ln, sn = values
            if not np.isfinite([lf, ln]).all() or lf <= 0 or ln <= 0:
                raise ValueError(f'{path}: invalid biaxial deformation at row {row+1}, ratio {ratio}')
            F = np.diag([lf, 1 / (lf * ln), ln])
            for suffix, stress, axis, stretch in (('f', sf, 0, lf), ('n', sn, 2, ln)):
                if np.isnan(stress):
                    continue
                if not np.isfinite(stress):
                    raise ValueError(f'{path}: nonfinite biaxial stress at row {row+1}')
                Fs.append(F)
                targets.append(stress)
                components.append((axis, axis))
                modes.append(f'{ratio}_{suffix}')
                groups.append(f'{ratio}:{row}')
                records.append({'mode': f'{ratio}_{suffix}', 'x': stretch, 'target': stress, 'row': row})
    observations = Observations(np.array(Fs), np.array(targets), tuple(components),
                                np.array(modes), np.array(groups), 'cauchy', 'kPa')
    return observations, pd.DataFrame(records)

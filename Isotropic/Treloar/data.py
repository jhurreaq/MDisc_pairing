"""Treloar and Kawabata nominal stresses in MPa; explicit, lossless CSV parsing."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_data(input_dir):
    input_dir = Path(input_dir)
    frames = []
    for mode in ('UT', 'PS', 'EBT'):
        path = input_dir / f'TreloarData{mode}.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, sep=';')
        if len(df.columns) != 2:
            raise ValueError(f'{path}: expected stretch and nominal stress')
        lam, stress = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
        l2 = lam ** (-0.5) if mode == 'UT' else (np.ones_like(lam) if mode == 'PS' else lam)
        frames.append(pd.DataFrame({'lambda1': lam, 'lambda2': l2, 'lambda3': 1/(lam*l2),
                                    'P11': stress, 'P22': stress if mode == 'EBT' else (0 if mode == 'UT' else np.nan),
                                    'mode': mode, 'dataset': 'Treloar'}))
    path = input_dir / 'Kawabata.csv'
    if path.is_file():
        # Header is row zero. Do not let the first zero-stress observation
        # become a header (the old skiprows=1 parser discarded this row).
        df = pd.read_csv(path, sep=None, engine='python')
        if len(df.columns) != 4:
            raise ValueError(f'{path}: expected lambda1, lambda2, P11, P22')
        df.columns = ['lambda1', 'lambda2', 'P11', 'P22']
        df = df.astype(float)
        df['lambda3'] = 1 / (df['lambda1'] * df['lambda2'])
        df['mode'], df['dataset'] = 'BIAX', 'Kawabata'
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)
    frame['strain_pct'] = (frame['lambda1'] - 1) * 100
    return frame

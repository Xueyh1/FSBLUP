"""Compare fit --write-k output to files from nearpd_official.R, with ID alignment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(path, **opts):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', **opts)


def magnitude(matrix):
    return float(np.max(np.abs(matrix)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference', type=Path)
    cli.add_argument('cpp', type=Path)
    args = cli.parse_args()
    ids = (args.reference / 'ids.txt').read_text().splitlines()
    r = np.fromfile(args.reference / 'nearPD.bin', dtype='<f8').reshape((len(ids), len(ids)), order='F')
    cpp = table(args.cpp / 'K.tsv', index_col=0)
    assert cpp.index.is_unique and cpp.columns.is_unique and len(ids) == len(set(ids)) == 1407
    assert set(ids) == set(cpp.index) == set(cpp.columns)
    k = cpp.loc[ids, ids].to_numpy()
    delta = k - r
    assert np.all(np.isfinite(k)) and np.array_equal(k, k.T)
    np.linalg.cholesky(k)
    kr = table(args.cpp / 'K_raw.tsv', index_col=0)
    assert list(kr.index) == list(cpp.index) and list(kr.columns) == list(cpp.columns)
    raw = kr.loc[ids,ids].to_numpy()
    raw_r = np.fromfile(args.reference/'raw.bin', dtype='<f8').reshape(raw.shape, order='F')
    pred = table(args.cpp / 'predictions.tsv', index_col='id')
    ref = table(args.reference / 'r_predictions.tsv', index_col='id')
    assert ref.index.is_unique and pred.index.is_unique and len(ref) == len(pred) == 198
    assert list(ref.index) == list(pred.index) == list(table(args.reference/"masked.tsv")["id"])
    ref = ref.loc[pred.index]
    candidates = pred['role'] == 'candidate'
    assert np.array_equal(np.flatnonzero(candidates), np.arange(0,198,10))
    info = table(args.cpp / 'model.tsv').iloc[0]
    output = {'n':len(k), 'max_abs_K':magnitude(delta),
              'relative_Frobenius_K': np.linalg.norm(delta) / np.linalg.norm(r),
              'repaired_K_min_eigen':float(np.linalg.eigvalsh(k)[0]),
              'max_abs_diag_K':magnitude(delta.diagonal()),
              'candidate_count':int(candidates.sum()), 'reference_count':int((~candidates).sum()),
              'cpp_lambda':float(info['lambda']), 'cpp_Vu':float(info['Vu']), 'cpp_Ve':float(info['Ve']),
              'cpp_fixed_intercept':float(table(args.cpp / 'fixed.tsv')['estimate'].iloc[0])}
    i,j = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    output.update(max_K_IDs=[ids[i],ids[j]], max_abs_raw_K=magnitude(raw-raw_r))
    without_jitter=k-raw
    shifted=raw.copy()
    shifted.flat[::len(raw)+1]+=1e-6
    output.update(max_abs_P_minus_raw=magnitude(without_jitter),
                  relative_Frobenius_P_minus_raw=float(np.linalg.norm(without_jitter)/np.linalg.norm(raw)),
                  relative_Frobenius_P_minus_S=float(np.linalg.norm(k-shifted)/np.linalg.norm(shifted)))
    held = table(args.reference/'heldout.tsv',index_col='id')
    assert list(held.index)==list(pred.index[candidates])
    yhat = pred.loc[held.index,'yhat']
    output.update(heldout_age_RMSE=float(np.sqrt(((held.age-yhat)**2).mean())),
                  heldout_age_Pearson=float(held.age.corr(yhat)))
    for column in ['u', 'yhat']:
        error = pred[column] - ref[column]
        output['max_abs_' + column] = magnitude(error)
        output['max_candidate_abs_' + column] = magnitude(error[candidates])
    r_model = table(args.reference / 'r_model.tsv').iloc[0]
    for column in ['Vu', 'Ve', 'lambda']:
        output['r_' + column] = float(r_model[column])
        output['abs_' + column] = abs(info[column] - r_model[column])
    output['abs_intercept'] = abs(output['cpp_fixed_intercept'] - r_model['intercept'])
    # Isolate the effect of the fitted variance ratio. Evaluate GLS/BLUP
    # with the C++ kernel and exactly the variance ratio returned by R.
    y = table(args.reference / 'masked.tsv', index_col='id').loc[pred.index, 'age']
    train = y.notna()
    assert np.array_equal(~train, candidates)
    k_subjects = cpp.loc[pred.index, pred.index].to_numpy()
    v = k_subjects[train][:, train] + r_model['lambda'] * np.eye(train.sum())
    observed = y[train].to_numpy()
    solved = np.linalg.solve(v, np.column_stack([np.ones(train.sum()), observed]))
    fixed = solved[:, 1].sum() / solved[:, 0].sum()
    u_at_r_lambda = k_subjects[:, train] @ (solved[:, 1] - fixed * solved[:, 0])
    output['max_abs_u_at_r_lambda'] = magnitude(u_at_r_lambda - ref['u'])
    output['max_abs_yhat_at_r_lambda'] = magnitude(fixed + u_at_r_lambda - ref['yhat'])
    vc = k_subjects[train][:, train] + info['lambda']*np.eye(train.sum())
    sc = np.linalg.solve(vc, np.column_stack([np.ones(train.sum()), observed]))
    bc = sc[:,1].sum()/sc[:,0].sum()
    uc = k_subjects[:,train] @ (sc[:,1]-bc*sc[:,0])
    output['max_abs_GLS_at_cpp_lambda'] = max(magnitude(uc-pred.u),magnitude(uc+bc-pred.yhat))
    assert output['max_abs_GLS_at_cpp_lambda']<1e-10 and output['max_abs_raw_K']<1e-11
    print(json.dumps(output, indent=2, allow_nan=False))
    assert output['max_abs_K'] < 1e-11
    assert output['relative_Frobenius_K'] < 1e-12
    assert max(output['max_abs_u'], output['max_abs_yhat']) < 1e-4
    assert max(output['max_abs_u_at_r_lambda'], output['max_abs_yhat_at_r_lambda']) < 1e-10


if __name__ == '__main__':
    main()

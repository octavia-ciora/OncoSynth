# %%
import numpy as np
import pandas as pd
import os
from importlib.metadata import PackageNotFoundError, version

os.environ["RPY2_CFFI_MODE"] = "ABI"
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

importr("grf")


def get_r_session_versions():
    """Return versions for the R session used by rpy2 and the grf package."""
    try:
        rpy2_version = version("rpy2")
    except PackageNotFoundError:
        rpy2_version = "unknown"

    return {
        "rpy2": rpy2_version,
        "R": ro.r("R.version.string")[0],
        "grf": ro.r('as.character(packageVersion("grf"))')[0],
        "grf_library_path": ro.r('system.file(package = "grf")')[0],
    }


def print_r_session_versions():
    versions = get_r_session_versions()
    print("R session versions:")
    for name, value in versions.items():
        print(f"  {name}: {value}")


import random

random.seed(42)
ro.r(
    """
options(grf.legacy.seed = FALSE)
set.seed(42)
"""
)


def _as_1d_int(x, name):
    x = np.asarray(x).reshape(-1)
    if not np.isin(x, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1. Got {np.unique(x)[:10]}")
    return x.astype(int)


def _as_1d_float(x, name):
    x = np.asarray(x).reshape(-1).astype(float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")
    return x


def csf_fit(
    X,
    W,
    T,
    C,
    horizon,
    num_trees=1000,
    min_node_size=15,
    sample_fraction=0.5,
    mtry=None,
    honesty=True,
    honesty_fraction=0.5,
    honesty_prune_leaves=True,
    alpha=0.05,
    imbalance_penalty=0.0,
    stabilize_splits=True,
    seed=42,
    num_threads=1,
):
    """Fit grf::causal_survival_forest and return an R model object (csf)."""
    horizon = float(horizon)

    if isinstance(X, np.ndarray):
        df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    elif isinstance(X, pd.DataFrame):
        df_X = X.copy()
    else:
        raise TypeError("X must be pandas DataFrame or numpy array.")

    W = _as_1d_int(W, "W")
    D = _as_1d_int(C, "C (event indicator -> D)")
    Y = _as_1d_float(T, "T (time -> Y)")

    n = df_X.shape[0]
    if len(W) != n or len(D) != n or len(Y) != n:
        raise ValueError("X, W, T, C must have the same length.")

    if mtry is None:
        p = df_X.shape[1]
        mtry = min(int(np.ceil(np.sqrt(p) + 20)), p)

    print(
        "Fitting CSF with "
        f"num_trees={num_trees}, "
        f"min_node_size={min_node_size}, "
        f"sample_fraction={sample_fraction}, "
        f"mtry={mtry}, "
        f"honesty={honesty}, "
        f"honesty_fraction={honesty_fraction}, "
        f"alpha={alpha}, "
        f"imbalance_penalty={imbalance_penalty}, "
        f"horizon={horizon}, "
        f"seed={seed}, "
        f"num_threads={num_threads}"
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["X_train"] = df_X

    ro.globalenv["W_train"] = ro.IntVector(W.tolist())
    ro.globalenv["Y_train"] = ro.FloatVector(Y.tolist())
    ro.globalenv["D_train"] = ro.IntVector(D.tolist())
    ro.globalenv["HORIZON"] = ro.FloatVector([horizon])

    ro.r(
        f"""
        options(grf.legacy.seed = FALSE)
        set.seed({int(seed)})
        
        csf <- grf::causal_survival_forest(
        X = as.matrix(X_train),
        W = W_train,
        Y = Y_train,
        D = D_train,
        horizon = HORIZON[1],
        num.trees = {int(num_trees)},
        sample.fraction = {float(sample_fraction)},
        mtry = {int(mtry)},
        min.node.size = {int(min_node_size)},
        honesty = {str(bool(honesty)).upper()},
        honesty.fraction = {float(honesty_fraction)},
        honesty.prune.leaves = {str(bool(honesty_prune_leaves)).upper()},
        alpha = {float(alpha)},
        imbalance.penalty = {float(imbalance_penalty)},
        stabilize.splits = {str(bool(stabilize_splits)).upper()},
        seed = {int(seed)},
        num.threads = {int(num_threads)}
        )
    """
    )

    return ro.globalenv["csf"]


def get_ate_from_csf(csf, wald_ci=False):
    """
    AIPW/orthogonal ATE computed on the same data the forest was trained on.
    """
    ro.globalenv["csf_obj"] = csf
    ate_r = ro.r("average_treatment_effect(csf_obj)")

    ate = {name: float(ate_r.rx2(name)[0]) for name in ate_r.names}

    if wald_ci and ("estimate" in ate) and ("std.err" in ate):
        est, se = ate["estimate"], ate["std.err"]
        ate["ci.lower.wald"] = est - 1.96 * se
        ate["ci.upper.wald"] = est + 1.96 * se

    return ate


def csf_predict_ite(csf, X_new, num_threads=1):
    """Predict ITE on new covariates using a fitted csf model."""
    if isinstance(X_new, np.ndarray):
        X_df = pd.DataFrame(X_new, columns=[f"x{i}" for i in range(X_new.shape[1])])
    elif isinstance(X_new, pd.DataFrame):
        X_df = X_new.copy()
    else:
        raise TypeError("X_new must be pandas DataFrame or numpy array.")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["X_new"] = X_df

    ro.globalenv["csf_obj"] = csf
    ite = np.asarray(
        ro.r(
            f"predict(csf_obj, newdata = as.matrix(X_new), num.threads = {int(num_threads)})$predictions"
        ),
        dtype=float,
    )
    return ite


def csf_variable_importance(csf, feature_names):
    ro.globalenv["csf_obj"] = csf
    vi = np.array(ro.r("grf::variable_importance(csf_obj)"), dtype=float).ravel()

    if len(vi) != len(feature_names):
        raise ValueError(
            f"Length mismatch: vi={len(vi)} vs feature_names={len(feature_names)}"
        )

    return pd.Series(vi, index=feature_names).sort_values(ascending=False)


if __name__ == "__main__":
    print_r_session_versions()

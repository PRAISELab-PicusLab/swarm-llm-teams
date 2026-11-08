"""
Sparse Group Lasso for Logistic Regression
===========================================

Implementation based on:
  - Yuan & Lin (2006)  "Model selection and estimation in regression
                        with grouped variables"
  - Simon et al. (2013) "A sparse-group lasso"

Solver: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
        with backtracking line search.

The penalty is:

    λ · [ α · Σ_j |β_j|  +  (1−α) · Σ_g √p_g · ‖β_g‖₂ ]

where g indexes groups (agents), p_g is group size, and α ∈ [0,1]
controls the balance between within-group sparsity (L1, LASSO-like)
and between-group sparsity (Group-L2, Group-LASSO-like).

  α → 1 : pure LASSO   (sparse features, ignores groups)
  α → 0 : pure Group LASSO (selects/excludes entire groups)
  α ≈ 0.3-0.5 : Sparse Group LASSO (sweet spot for our pipeline)
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.validation import check_is_fitted


# ====================================================================
# Helpers
# ====================================================================

def build_group_indices(
    feature_cols: list[str],
    feature_to_agent: dict[str, str],
) -> np.ndarray:
    """Convert the feature → agent mapping to a numeric group-index array.

    Returns
    -------
    groups : np.ndarray of shape (n_features,)
        Integer group label for each feature (e.g. [0,0,0,0, 1,1,1, 2,2, ...]).
    agent_order : list[str]
        The agent name corresponding to each numeric group index.
    """
    agents_seen: list[str] = []
    agent_to_idx: dict[str, int] = {}
    groups = np.empty(len(feature_cols), dtype=int)

    for i, feat in enumerate(feature_cols):
        agent = feature_to_agent.get(feat, f"UNKNOWN_{feat}")
        if agent not in agent_to_idx:
            agent_to_idx[agent] = len(agents_seen)
            agents_seen.append(agent)
        groups[i] = agent_to_idx[agent]

    return groups, agents_seen


def _compute_group_weights(groups: np.ndarray) -> np.ndarray:
    """√p_g for each group (correction for different group sizes)."""
    unique_g = np.unique(groups)
    weights = np.zeros(unique_g.max() + 1, dtype=float)
    for g in unique_g:
        weights[g] = np.sqrt(np.sum(groups == g))
    return weights


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute sample weights equivalent to class_weight='balanced'."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    w = np.ones(n, dtype=float)
    for c, cnt in zip(classes, counts):
        w[y == c] = n / (n_classes * cnt)
    return w


# ====================================================================
# Proximal operator for SGL
# ====================================================================

def _prox_sgl(
    v: np.ndarray,
    lam: float,
    alpha: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    step: float,
) -> np.ndarray:
    """Two-stage proximal operator for Sparse Group Lasso.

    Stage 1 – Soft-thresholding (L1, feature-level):
        z_j = sign(v_j) · max(|v_j| − α·λ·step, 0)

    Stage 2 – Group soft-thresholding (L2, group-level):
        β_g = z_g · max(1 − (1−α)·λ·step·√p_g / ‖z_g‖₂ , 0)
    """
    # Stage 1: element-wise soft-threshold
    threshold_l1 = alpha * lam * step
    z = np.sign(v) * np.maximum(np.abs(v) - threshold_l1, 0.0)

    # Stage 2: group soft-threshold
    beta = z.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        z_g = z[idx]
        norm_g = np.linalg.norm(z_g)
        if norm_g < 1e-15:
            beta[idx] = 0.0
        else:
            threshold_grp = (1.0 - alpha) * lam * step * group_weights[g]
            shrink = max(1.0 - threshold_grp / norm_g, 0.0)
            beta[idx] = z_g * shrink

    return beta


# ====================================================================
# Sigmoid & loss utilities
# ====================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _weighted_log_loss(X, y, beta, intercept, sample_weight):
    """Weighted negative log-likelihood (to minimize)."""
    z = X @ beta + intercept
    p = _sigmoid(z)
    # Numerically stable log-loss
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return np.sum(sample_weight * ll) / np.sum(sample_weight)


def _gradient(X, y, beta, intercept, sample_weight):
    """Gradient of the weighted log-loss w.r.t. beta and intercept."""
    n = len(y)
    z = X @ beta + intercept
    p = _sigmoid(z)
    residual = sample_weight * (p - y)  # shape (n,)
    w_sum = np.sum(sample_weight)
    grad_beta = (X.T @ residual) / w_sum
    grad_intercept = np.sum(residual) / w_sum
    return grad_beta, grad_intercept


# ====================================================================
# FISTA solver
# ====================================================================

def _fit_sgl_fista(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    sample_weight: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, dict]:
    """Solve SGL-penalized logistic regression via FISTA.

    Returns (beta, intercept, info_dict).
    """
    n, p = X.shape

    # Initial step size: 1/L where L upper-bounds the Lipschitz constant.
    # For weighted logistic loss: L ≤ ‖X‖²_op · max(w) / (4 · sum(w))
    # We use Frobenius norm as cheap upper bound of spectral norm.
    w_max = np.max(sample_weight)
    w_sum = np.sum(sample_weight)
    L0 = (np.linalg.norm(X, ord="fro") ** 2) * w_max / (4.0 * w_sum)
    L0 = max(L0, 1e-6)
    step = 1.0 / L0

    # Initialize
    beta = np.zeros(p, dtype=float)
    intercept = 0.0

    # FISTA momentum variables
    beta_prev = beta.copy()
    intercept_prev = intercept
    t_fista = 1.0

    converged = False
    for iteration in range(max_iter):
        # FISTA extrapolation point
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_fista ** 2)) / 2.0
        momentum = (t_fista - 1.0) / t_new

        beta_ext = beta + momentum * (beta - beta_prev)
        intercept_ext = intercept + momentum * (intercept - intercept_prev)

        # Gradient at extrapolation point
        g_beta, g_intercept = _gradient(X, y, beta_ext, intercept_ext, sample_weight)

        # Backtracking line search
        current_step = step * 2.0  # start optimistic
        for _bt in range(30):
            # Gradient step
            v = beta_ext - current_step * g_beta
            b_new = intercept_ext - current_step * g_intercept

            # Proximal step (only on beta, intercept is unpenalized)
            beta_cand = _prox_sgl(v, lam, alpha, groups, group_weights, current_step)

            # Check sufficient decrease (Armijo-like)
            loss_cand = _weighted_log_loss(X, y, beta_cand, b_new, sample_weight)
            loss_ext = _weighted_log_loss(X, y, beta_ext, intercept_ext, sample_weight)

            diff = beta_cand - beta_ext
            quadratic_bound = (
                loss_ext
                + np.dot(g_beta, diff)
                + (0.5 / current_step) * np.dot(diff, diff)
            )
            if loss_cand <= quadratic_bound + 1e-12:
                break
            current_step *= 0.5

        step = current_step  # remember for next iteration

        # Update
        beta_prev = beta.copy()
        intercept_prev = intercept
        beta = beta_cand
        intercept = b_new
        t_fista = t_new

        # Convergence check: relative change in beta
        change = np.linalg.norm(beta - beta_prev)
        norm_beta = np.linalg.norm(beta) + 1e-12
        if change / norm_beta < tol and iteration > 5:
            converged = True
            break

    info = {
        "converged": converged,
        "n_iter": iteration + 1,
        "final_loss": float(_weighted_log_loss(X, y, beta, intercept, sample_weight)),
    }

    return beta, intercept, info


# ====================================================================
# sklearn-compatible classifier
# ====================================================================

class SparseGroupLassoLogistic(BaseEstimator, ClassifierMixin):
    """Sparse Group Lasso for binary logistic regression.

    Parameters
    ----------
    lam : float
        Regularization strength (higher = more sparse).
    alpha : float in [0, 1]
        Mixing parameter.  α=1 → pure LASSO, α=0 → pure Group LASSO.
    groups : array-like of int
        Group label for each feature (e.g. [0,0,0,0, 1,1,1, 2,2, ...]).
    class_weight : str or None
        If 'balanced', compute sample weights inversely proportional to class freq.
    max_iter : int
        Maximum FISTA iterations.
    tol : float
        Convergence tolerance.
    random_state : int or None
        Not used by solver (deterministic), kept for API compatibility.
    """

    def __init__(
        self,
        lam: float = 0.1,
        alpha: float = 0.3,
        groups: np.ndarray | None = None,
        class_weight: str | None = "balanced",
        max_iter: int = 5000,
        tol: float = 1e-7,
        random_state: int | None = None,
    ):
        self.lam = lam
        self.alpha = alpha
        self.groups = groups
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        # Groups
        if self.groups is None:
            self.groups_ = np.arange(p)  # each feature is its own group
        else:
            self.groups_ = np.asarray(self.groups, dtype=int)
        self.group_weights_ = _compute_group_weights(self.groups_)

        # Sample weights
        if self.class_weight == "balanced":
            sw = _balanced_sample_weights(y)
        else:
            sw = np.ones(n, dtype=float)

        # Solve
        beta, intercept, info = _fit_sgl_fista(
            X, y,
            lam=self.lam,
            alpha=self.alpha,
            groups=self.groups_,
            group_weights=self.group_weights_,
            sample_weight=sw,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # Store in sklearn-compatible format
        self.coef_ = beta.reshape(1, -1)       # shape (1, p)
        self.intercept_ = np.array([intercept]) # shape (1,)
        self.classes_ = np.array([0, 1])
        self.fit_info_ = info

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        check_is_fitted(self)
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        check_is_fitted(self)
        return (self.decision_function(X) >= 0.0).astype(int)


# ====================================================================
# Cross-validation wrapper
# ====================================================================

# Default grids -------------------------------------------------------
DEFAULT_LAMS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


class SparseGroupLassoCV:
    """Cross-validated Sparse Group Lasso for binary logistic regression.

    Searches over a (λ, α) grid using stratified K-fold.

    Selection rule (coverage-aware SE):
      1. Find the best mean balanced_accuracy across all (λ, α) combos.
      2. Identify all combos within `se_multiplier` standard errors of the best.
      3. Among those, pick the combo that activates the most groups.
      4. Ties in group count broken by lower α (more group structure).

    A higher `se_multiplier` widens the candidate window, increasing the
    chance that the selected model covers more agent groups — at the cost
    of a potentially lower predictive score.  The default (1.0) reproduces
    the classical 1-SE rule (Hastie, Tibshirani & Friedman, 2009).

    After selection the model is refit on the full training data.

    Attributes (after fit)
    ----------------------
    best_lam_ : float
    best_alpha_ : float
    best_score_ : float
    coef_ : ndarray, shape (1, p)
    intercept_ : ndarray, shape (1,)
    cv_results_ : list[dict]   (all evaluated combos)
    """

    def __init__(
        self,
        lams: list[float] | None = None,
        alphas: list[float] | None = None,
        groups: np.ndarray | None = None,
        cv: int = 5,
        scoring: str = "balanced_accuracy",
        class_weight: str | None = "balanced",
        max_iter: int = 5000,
        tol: float = 1e-7,
        random_state: int | None = None,
        se_multiplier: float = 1.0,
    ):
        self.lams = lams if lams is not None else DEFAULT_LAMS
        self.alphas = alphas if alphas is not None else DEFAULT_ALPHAS
        self.groups = groups
        self.cv = cv
        self.scoring = scoring
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.se_multiplier = se_multiplier

    # --- internal scorer ----
    @staticmethod
    def _score(clf, X_val, y_val) -> float:
        y_pred = clf.predict(X_val)
        return balanced_accuracy_score(y_val, y_pred)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Resolve groups
        groups = np.arange(X.shape[1]) if self.groups is None else np.asarray(self.groups, dtype=int)
        n_total_groups = len(np.unique(groups))

        # Prepare CV splitter
        n_splits = self.cv
        counts = np.unique(y, return_counts=True)[1]
        min_count = int(counts.min())
        n_splits = max(2, min(n_splits, min_count))

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Grid search
        results: list[dict] = []

        for lam in self.lams:
            for alpha in self.alphas:
                fold_scores = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    clf = SparseGroupLassoLogistic(
                        lam=lam,
                        alpha=alpha,
                        groups=groups,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state,
                    )
                    try:
                        clf.fit(X_tr, y_tr)
                        fold_scores.append(self._score(clf, X_val, y_val))
                    except Exception:
                        fold_scores.append(0.0)

                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))

                results.append({
                    "lam": lam,
                    "alpha": alpha,
                    "mean_balanced_accuracy": mean_score,
                    "std_balanced_accuracy": std_score,
                    "fold_scores": fold_scores,
                })

        # ── Coverage-aware 1-SE selection rule ──
        # Step 1: find the best mean score and its SE
        best_mean = max(r["mean_balanced_accuracy"] for r in results)
        best_entry = next(r for r in results if r["mean_balanced_accuracy"] == best_mean)
        best_se = best_entry["std_balanced_accuracy"] / np.sqrt(n_splits)

        # Step 2: candidates within se_multiplier * SE of best
        threshold = best_mean - self.se_multiplier * best_se
        candidates = [r for r in results if r["mean_balanced_accuracy"] >= threshold]

        # Step 3: for each candidate, quick refit on full data to count active groups
        for cand in candidates:
            tmp_clf = SparseGroupLassoLogistic(
                lam=cand["lam"],
                alpha=cand["alpha"],
                groups=groups,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            tmp_clf.fit(X, y)
            coef_tmp = tmp_clf.coef_[0]
            nz = np.abs(coef_tmp) > 1e-8
            cand["n_active_groups"] = len(set(groups[j] for j in range(len(groups)) if nz[j]))
            cand["n_nonzero"] = int(nz.sum())

        # Step 4: sort candidates by (most groups, lowest α, highest score)
        candidates.sort(key=lambda r: (
            -r["n_active_groups"],        # more groups is better
            r["alpha"],                   # lower α → more group structure
            -r["mean_balanced_accuracy"], # higher score as tiebreak
        ))

        chosen = candidates[0]
        best_lam = chosen["lam"]
        best_alpha = chosen["alpha"]
        best_score = chosen["mean_balanced_accuracy"]

        n_within_1se = len(candidates)
        print(
            f"[SGL-CV] {self.se_multiplier}-SE rule: {n_within_1se} candidates within "
            f"{self.se_multiplier}-SE (threshold={threshold:.3f}). "
            f"Chosen: λ={best_lam:.4f}, α={best_alpha:.2f} "
            f"(groups={chosen.get('n_active_groups', '?')}, "
            f"score={best_score:.3f})"
        )

        # Refit on full data with chosen hyperparameters
        self.best_estimator_ = SparseGroupLassoLogistic(
            lam=best_lam,
            alpha=best_alpha,
            groups=groups,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.best_estimator_.fit(X, y)

        # Expose attributes for pipeline compatibility
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.classes_ = self.best_estimator_.classes_

        self.best_lam_ = best_lam
        self.best_alpha_ = best_alpha
        self.best_score_ = best_score
        self.cv_results_ = results

        n_nz = int(np.sum(np.abs(self.coef_[0]) > 1e-8))
        n_groups_active = len(set(
            groups[j] for j in range(len(groups))
            if abs(self.coef_[0][j]) > 1e-8
        ))
        print(
            f"[SGL-CV] Best λ={best_lam:.4f}, α={best_alpha:.2f}, "
            f"bal_acc={best_score:.3f}, "
            f"non-zero={n_nz}/{X.shape[1]}, "
            f"active groups={n_groups_active}/{n_total_groups}"
        )

        return self

    # Delegate predict/predict_proba to best estimator
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)

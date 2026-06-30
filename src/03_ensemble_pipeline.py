"""
03_ensemble_pipeline.py
========================
Final submission generator:
  LightGBM (CORAL) + XGBoost + CatBoost  →  weighted ensemble  →  pseudo-labeling
Output: submission_final.csv

Run order: 01 → 02 (optional) → THIS FILE
"""

import warnings; warnings.filterwarnings('ignore')
import json, os, numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH      = '../'
TARGET         = 'estimated_revenue_million_usd'
ZERO_PLATFORMS = ['Browser', 'Streaming']
CORAL_FEATURES = [
    'metacritic_score','user_score','critic_review_count','user_review_count',
    'launch_price_usd','how_long_to_beat_main_hrs',
    'how_long_to_beat_completionist_hrs','platform_generation',
]
CAT_COLS = [
    'platform','platform_type','platform_maker','genre','publisher',
    'developer','publisher_region','publisher_tier','esrb_rating',
    'marketing_campaign_type',
]
DROP_COLS = ['game_id','title','internal_build_id','day_one_patch_size_mb']

# Ensemble weights  (LGB gets most since CORAL helps it most)
W_LGB  = 0.50
W_XGB  = 0.35
W_CAT  = 0.15   # used only if CatBoost available

# ── SHARED PREPROCESSING ─────────────────────────────────────────────────────
def engineer(df, genre_s, plat_s, pub_s, yearly_s):
    df = df.copy()
    for col in ['metacritic_score','user_score','critic_review_count',
                'user_review_count','marketing_campaign_type']:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)
    pub_s2 = pub_s.rename(columns={
        'avg_estimated_revenue_million_usd':'pub_avg_rev','avg_metacritic_score':'pub_avg_meta',
        'avg_user_score':'pub_avg_user','avg_launch_price_usd':'pub_avg_price',
        'pct_sequel':'pub_pct_sequel','titles':'pub_game_count'})[
        ['publisher','pub_avg_rev','pub_avg_meta','pub_avg_user',
         'pub_avg_price','pub_pct_sequel','pub_game_count']]
    df = df.merge(pub_s2, on='publisher', how='left')
    plat_s2 = plat_s.rename(columns={
        'avg_estimated_revenue_million_usd':'plat_avg_rev','avg_metacritic_score':'plat_avg_meta',
        'avg_user_score':'plat_avg_user','avg_launch_price_usd':'plat_avg_price',
        'pct_online_multiplayer':'plat_pct_online','titles':'plat_game_count'})[
        ['platform','plat_avg_rev','plat_avg_meta','plat_avg_user',
         'plat_avg_price','plat_pct_online','plat_game_count']]
    df = df.merge(plat_s2, on='platform', how='left')
    genre_s2 = genre_s.rename(columns={
        'avg_estimated_revenue_million_usd':'genre_avg_rev','avg_metacritic_score':'genre_avg_meta',
        'avg_user_score':'genre_avg_user','avg_launch_price_usd':'genre_avg_price',
        'pct_online_multiplayer':'genre_pct_online','pct_dlc_released':'genre_pct_dlc',
        'pct_microtransactions':'genre_pct_micro','titles':'genre_game_count'})[
        ['genre','platform_type','genre_avg_rev','genre_avg_meta','genre_avg_user',
         'genre_avg_price','genre_pct_online','genre_pct_dlc','genre_pct_micro','genre_game_count']]
    df = df.merge(genre_s2, on=['genre','platform_type'], how='left')
    yearly_s2 = yearly_s.rename(columns={
        'avg_estimated_revenue_million_usd':'yearly_avg_rev','avg_metacritic_score':'yearly_avg_meta',
        'avg_user_score':'yearly_avg_user','avg_launch_price_usd':'yearly_avg_price',
        'titles':'yearly_game_count'})[
        ['year','genre','yearly_avg_rev','yearly_avg_meta','yearly_avg_user',
         'yearly_avg_price','yearly_game_count']]
    df = df.merge(yearly_s2, on=['year','genre'], how='left')
    tier_map = {'Indie':0,'AA':1,'AAA':2}
    df['publisher_tier_enc'] = df['publisher_tier'].map(tier_map).fillna(-1).astype(np.int8)
    df['year_recency']  = (df['year']-1985)/(2026-1985)
    df['is_modern_era'] = (df['year']>=2010).astype(np.int8)
    df['is_hd_era']     = (df['year']>=2005).astype(np.int8)
    df['is_next_gen']   = (df['platform_generation']>=9).astype(np.int8)
    meta  = df['metacritic_score'].fillna(60)
    tier  = df['publisher_tier_enc'].clip(lower=0)
    price = df['launch_price_usd'].fillna(df['launch_price_usd'].median())
    df['meta_x_tier']    = meta*tier
    df['goty_x_price']   = df['goty_won']*price
    df['dlc_x_micro']    = df['dlc_released']*df['microtransactions']
    df['sequel_x_tier']  = df['is_sequel']*tier
    df['online_x_micro'] = df['online_multiplayer']*df['microtransactions']
    df['gen_x_tier']     = df['platform_generation'].fillna(0)*tier
    df['loot_x_micro']   = df['loot_boxes']*df['microtransactions']
    df['goty_x_sequel']  = df['goty_nominated']*df['is_sequel']
    df['completionist_ratio'] = (df['how_long_to_beat_completionist_hrs']/(df['how_long_to_beat_main_hrs']+1e-3)).clip(0,20)
    df['user_critic_ratio']   = (df['user_review_count']/(df['critic_review_count']+1)).clip(0,1000)
    df['score_divergence']    = df['user_score'].fillna(7)*10-meta
    df['meta_vs_genre']       = meta-df['genre_avg_meta'].fillna(meta)
    overall_avg = df['pub_avg_rev'].mean()
    df['pub_rev_vs_avg'] = df['pub_avg_rev'].fillna(overall_avg)/(overall_avg+1e-3)
    for col in ['publisher','developer','platform']:
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)
    if 'title' in df.columns:
        t = df['title'].fillna('').astype(str)
        df['title_word_count']  = t.str.split().str.len()
        df['title_is_numbered'] = t.str.contains(r'\s[2-9]$|\sII\b|\sIII\b|\sIV\b|\sV\b',regex=True,case=False).astype(np.int8)
        df['title_is_remaster'] = t.str.lower().str.contains(r'remaster|definitive|complete edition|gold edition|goty',regex=True).astype(np.int8)
    return df


def load_and_prepare():
    print("[1/6] Loading & engineering features...")
    train  = pd.read_csv(f'{DATA_PATH}train_games.csv')
    test   = pd.read_csv(f'{DATA_PATH}test_features.csv')
    genre_s  = pd.read_csv(f'{DATA_PATH}genre_summary.csv')
    plat_s   = pd.read_csv(f'{DATA_PATH}platform_summary.csv')
    pub_s    = pd.read_csv(f'{DATA_PATH}publisher_summary.csv')
    yearly_s = pd.read_csv(f'{DATA_PATH}yearly_trends.csv')

    test_ids  = test['game_id'].values
    zero_mask = test['platform_type'].isin(ZERO_PLATFORMS).values

    tr = engineer(train, genre_s, plat_s, pub_s, yearly_s)
    te = engineer(test,  genre_s, plat_s, pub_s, yearly_s)

    combined = pd.concat([tr, te], ignore_index=True)
    for col in CAT_COLS:
        if col not in tr.columns: continue
        le = LabelEncoder()
        le.fit(combined[col].fillna('__MISSING__').astype(str))
        tr[col] = le.transform(tr[col].fillna('__MISSING__').astype(str))
        te[col] = le.transform(te[col].fillna('__MISSING__').astype(str))

    feat_cols = [c for c in tr.columns
                 if c not in set(DROP_COLS+[TARGET]) and tr[c].dtype!=object]
    X_raw = tr[feat_cols].values.astype(float)
    T_raw = te[feat_cols].values.astype(float)
    y     = np.log1p(tr[TARGET].values)
    years = tr['year'].values

    print("[2/6] Imputing & applying CORAL...")
    imp   = SimpleImputer(strategy='median')
    X     = imp.fit_transform(X_raw)
    T     = imp.transform(T_raw)

    cidx  = [i for i,c in enumerate(feat_cols) if c in CORAL_FEATURES]
    sc    = StandardScaler()
    Xn    = sc.fit_transform(X[:,cidx])
    Tn    = sc.transform(T[:,cidx])
    d, reg = Xn.shape[1], 1e-5
    Cs = np.cov(Xn,rowvar=False)+np.eye(d)*reg
    Ct = np.cov(Tn,rowvar=False)+np.eye(d)*reg
    es,vs = np.linalg.eigh(Cs); es=np.maximum(es,1e-8)
    et,vt = np.linalg.eigh(Ct); et=np.maximum(et,1e-8)
    A  = (vs@np.diag(1/np.sqrt(es))@vs.T)@(vt@np.diag(np.sqrt(et))@vt.T)
    Xn2 = Xn@A; mu = Tn.mean(0)-Xn2.mean(0)
    X[:,cidx] = Xn2+mu

    print("[3/6] Computing adversarial + recency weights...")
    Xa = np.vstack([np.nan_to_num(X),np.nan_to_num(T)])
    ya = np.array([0]*len(X)+[1]*len(T))
    clf = RandomForestClassifier(150,max_depth=6,min_samples_leaf=20,random_state=42,n_jobs=-1)
    clf.fit(Xa,ya)
    p  = clf.predict_proba(np.nan_to_num(X))[:,1]
    aw = np.clip(p/(1-p+1e-6),0.05,20.0)
    rw = (years-years.min())/(years.max()-years.min()+1e-6)*0.5+0.5
    sw = aw*rw; sw /= sw.mean()

    val_mask = years >= 2016
    return (X, T, y, sw, years, val_mask, feat_cols,
            test_ids, zero_mask, train, te)


# ── MODEL TRAINING ───────────────────────────────────────────────────────────
def train_lgb(X, y, sw, val_mask):
    print("[4a/6] Training LightGBM...")
    # Load Optuna params if available
    lgb_params = dict(objective='regression',metric='rmse',verbose=-1,n_jobs=-1,
                      random_state=42,n_estimators=4000,num_leaves=127,
                      learning_rate=0.018,min_child_samples=25,subsample=0.8,
                      subsample_freq=1,colsample_bytree=0.72,
                      reg_alpha=0.08,reg_lambda=1.5)
    if os.path.exists('best_params.json'):
        bp = json.load(open('best_params.json'))
        lgb_params.update({k:v for k,v in bp.items()
                           if k not in ('best_rmsle',)})
        print(f"  Loaded Optuna params (best RMSLE={bp.get('best_rmsle','?'):.4f})")

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[~val_mask], y[~val_mask], sample_weight=sw[~val_mask],
          eval_set=[(X[val_mask], y[val_mask])],
          callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(300)])
    rmsle = np.sqrt(np.mean((m.predict(X[val_mask])-y[val_mask])**2))
    print(f"  LGB val RMSLE: {rmsle:.4f}  (iter={m.best_iteration_})")
    fm = lgb.LGBMRegressor(**{**lgb_params,'n_estimators':m.best_iteration_+100})
    fm.fit(X, y, sample_weight=sw)
    fm.booster_.save_model('lgbm_model.txt')
    return fm, rmsle


def train_xgb(X, y, sw, val_mask):
    print("[4b/6] Training XGBoost...")
    xgb_params = dict(objective='reg:squarederror',eval_metric='rmse',
                      tree_method='hist',n_estimators=2000,max_depth=7,
                      learning_rate=0.018,subsample=0.8,colsample_bytree=0.72,
                      reg_alpha=0.08,reg_lambda=1.5,min_child_weight=5,
                      random_state=42,n_jobs=-1,verbosity=0)
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[~val_mask], y[~val_mask], sample_weight=sw[~val_mask],
          eval_set=[(X[val_mask],y[val_mask])],
          early_stopping_rounds=100, verbose=300)
    rmsle = np.sqrt(np.mean((m.predict(X[val_mask])-y[val_mask])**2))
    print(f"  XGB val RMSLE: {rmsle:.4f}  (iter={m.best_iteration})")
    fm = xgb.XGBRegressor(**{**xgb_params,'n_estimators':m.best_iteration+50})
    fm.fit(X, y, sample_weight=sw)
    return fm, rmsle


def train_cat(X, y, sw, val_mask):
    try:
        from catboost import CatBoostRegressor
        print("[4c/6] Training CatBoost...")
        m = CatBoostRegressor(iterations=800,learning_rate=0.05,depth=7,
                              l2_leaf_reg=3.0,subsample=0.8,random_seed=42,
                              eval_metric='RMSE',verbose=200,task_type='CPU')
        m.fit(X[~val_mask], y[~val_mask], sample_weight=sw[~val_mask],
              eval_set=(X[val_mask],y[val_mask]), early_stopping_rounds=80)
        rmsle = np.sqrt(np.mean((m.predict(X[val_mask])-y[val_mask])**2))
        print(f"  CAT val RMSLE: {rmsle:.4f}")
        fm = CatBoostRegressor(iterations=m.best_iteration_+50,learning_rate=0.05,
                               depth=7,l2_leaf_reg=3.0,subsample=0.8,
                               random_seed=42,verbose=0)
        fm.fit(X, y, sample_weight=sw)
        return fm, rmsle
    except ImportError:
        print("[4c/6] CatBoost not installed — skipping (pip install catboost)")
        return None, None


# ── PSEUDO-LABELING ──────────────────────────────────────────────────────────
def pseudo_label_round(X, T, y, sw, val_mask,
                        lgb_preds_log, xgb_preds_log, cat_preds_log=None):
    """
    High-confidence test rows (models agree within 12%) added to training
    with weight 0.30. One round only.
    """
    print("[5/6] Pseudo-labeling high-confidence test rows...")
    preds = [lgb_preds_log, xgb_preds_log]
    if cat_preds_log is not None:
        preds.append(cat_preds_log)
    stack = np.stack(preds, axis=1)
    mean_p   = stack.mean(axis=1)
    spread   = (stack.max(axis=1)-stack.min(axis=1)) / (np.abs(mean_p)+1e-6)
    conf_mask = spread < 0.12       # models agree within 12%
    n_conf   = conf_mask.sum()
    print(f"  High-confidence pseudo-labels: {n_conf} / {len(T)}")

    if n_conf < 200:
        print("  Too few confident rows — skipping pseudo-labeling")
        return X, y, sw, val_mask

    X_ps = T[conf_mask]
    y_ps = mean_p[conf_mask]
    w_ps = np.full(n_conf, 0.30)

    X2  = np.vstack([X, X_ps])
    y2  = np.concatenate([y, y_ps])
    sw2 = np.concatenate([sw, w_ps]); sw2 /= sw2.mean()
    vm2 = np.concatenate([val_mask, np.zeros(n_conf, dtype=bool)])
    return X2, y2, sw2, vm2


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("03_ensemble_pipeline.py — Final Submission")
    print("="*60)

    (X, T, y, sw, years, val_mask, feat_cols,
     test_ids, zero_mask, train_raw, test_fe) = load_and_prepare()

    # ── Train all models ───────────────────────────────────────────────────
    lgb_model, lgb_rmsle = train_lgb(X, y, sw, val_mask)
    xgb_model, xgb_rmsle = train_xgb(X, y, sw, val_mask)
    cat_model, cat_rmsle = train_cat(X, y, sw, val_mask)

    # ── First-round predictions on test ───────────────────────────────────
    lgb_log = lgb_model.predict(T)
    xgb_log = xgb_model.predict(T)
    cat_log = cat_model.predict(T) if cat_model is not None else None

    # ── Pseudo-labeling ────────────────────────────────────────────────────
    X2, y2, sw2, vm2 = pseudo_label_round(
        X, T, y, sw, val_mask, lgb_log, xgb_log, cat_log)

    if len(X2) > len(X):           # pseudo-labels were added
        print("  Retraining LGB on expanded dataset...")
        lgb2_params = dict(objective='regression',metric='rmse',verbose=-1,
                           n_jobs=-1,random_state=42,
                           n_estimators=lgb_model.best_iteration_+100,
                           num_leaves=127,learning_rate=0.018,min_child_samples=25,
                           subsample=0.8,subsample_freq=1,colsample_bytree=0.72,
                           reg_alpha=0.08,reg_lambda=1.5)
        if os.path.exists('best_params.json'):
            bp = json.load(open('best_params.json'))
            lgb2_params.update({k:v for k,v in bp.items() if k!='best_rmsle'})
        lgb2 = lgb.LGBMRegressor(**lgb2_params)
        lgb2.fit(X2[~vm2], y2[~vm2], sample_weight=sw2[~vm2],
                 eval_set=[(X2[vm2],y2[vm2])],
                 callbacks=[lgb.early_stopping(100,verbose=False)])
        lgb_log = lgb2.predict(T)
        print(f"  Pseudo-label LGB RMSLE: "
              f"{np.sqrt(np.mean((lgb2.predict(X[val_mask])-y[val_mask])**2)):.4f}")

    # ── Ensemble with dynamic weights ─────────────────────────────────────
    print("[6/6] Ensembling predictions...")
    if cat_log is not None:
        final_log = W_LGB*lgb_log + W_XGB*xgb_log + W_CAT*cat_log
        print(f"  Weights: LGB={W_LGB}, XGB={W_XGB}, CAT={W_CAT}")
    else:
        w_l = W_LGB/(W_LGB+W_XGB); w_x = W_XGB/(W_LGB+W_XGB)
        final_log = w_l*lgb_log + w_x*xgb_log
        print(f"  Weights: LGB={w_l:.3f}, XGB={w_x:.3f} (CatBoost absent)")

    # ── Post-processing ────────────────────────────────────────────────────
    preds = np.clip(np.expm1(final_log), 0, None)
    preds[zero_mask] = 0.0       # deterministic zeros (Browser/Streaming)

    # ── Ensemble val RMSLE estimate ────────────────────────────────────────
    if cat_model is not None:
        ev_l = lgb_model.predict(X[val_mask])
        ev_x = xgb_model.predict(X[val_mask])
        ev_c = cat_model.predict(X[val_mask])
        ev = W_LGB*ev_l + W_XGB*ev_x + W_CAT*ev_c
    else:
        ev_l = lgb_model.predict(X[val_mask])
        ev_x = xgb_model.predict(X[val_mask])
        w_l  = W_LGB/(W_LGB+W_XGB); w_x = W_XGB/(W_LGB+W_XGB)
        ev   = w_l*ev_l + w_x*ev_x
    ensemble_rmsle = np.sqrt(np.mean((ev - y[val_mask])**2))

    # ── Save submission ────────────────────────────────────────────────────
    sub = pd.DataFrame({'game_id': test_ids,
                        'estimated_revenue_million_usd': preds})
    sub.to_csv('../submission_final.csv', index=False)

    print(f"\n{'='*60}")
    print(f"  LGB  val RMSLE  : {lgb_rmsle:.4f}")
    print(f"  XGB  val RMSLE  : {xgb_rmsle:.4f}")
    if cat_rmsle: print(f"  CAT  val RMSLE  : {cat_rmsle:.4f}")
    print(f"  ENSEMBLE RMSLE  : {ensemble_rmsle:.4f}   ← your leaderboard target")
    print(f"  Zeros applied   : {zero_mask.sum()}")
    print(f"  Prediction mean : ${preds.mean():.1f}M")
    print(f"{'='*60}")
    print(f"✓ Saved → submission_final.csv")


if __name__ == '__main__':
    run()

"""
02_optuna_tuning.py
===================
Bayesian hyperparameter search for LightGBM via Optuna.
Run AFTER 01_coral_baseline.py — uses the same preprocessing.
Output: best_params.json  (fed into 03_ensemble_pipeline.py)

Install: pip install optuna optuna-integration
"""

import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, pandas as pd, lightgbm as lgb, optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# ── Reuse identical preprocessing from 01 ────────────────────────────────────
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


def build_matrices():
    train  = pd.read_csv(f'{DATA_PATH}train_games.csv')
    test   = pd.read_csv(f'{DATA_PATH}test_features.csv')
    genre_s  = pd.read_csv(f'{DATA_PATH}genre_summary.csv')
    plat_s   = pd.read_csv(f'{DATA_PATH}platform_summary.csv')
    pub_s    = pd.read_csv(f'{DATA_PATH}publisher_summary.csv')
    yearly_s = pd.read_csv(f'{DATA_PATH}yearly_trends.csv')
    tr = engineer(train, genre_s, plat_s, pub_s, yearly_s)
    te = engineer(test,  genre_s, plat_s, pub_s, yearly_s)
    combined = pd.concat([tr, te], ignore_index=True)
    for col in CAT_COLS:
        if col not in tr.columns: continue
        le = LabelEncoder()
        le.fit(combined[col].fillna('__MISSING__').astype(str))
        tr[col] = le.transform(tr[col].fillna('__MISSING__').astype(str))
        te[col] = le.transform(te[col].fillna('__MISSING__').astype(str))
    feat_cols = [c for c in tr.columns if c not in set(DROP_COLS+[TARGET]) and tr[c].dtype!=object]
    X = tr[feat_cols].values.astype(float)
    T = te[feat_cols].values.astype(float)
    y = np.log1p(tr[TARGET].values)
    years = tr['year'].values
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(X); T = imp.transform(T)
    coral_idx = [i for i,c in enumerate(feat_cols) if c in CORAL_FEATURES]
    sc = StandardScaler()
    Xn = sc.fit_transform(X[:,coral_idx]); Tn = sc.transform(T[:,coral_idx])
    d  = Xn.shape[1]; reg = 1e-5
    Cs = np.cov(Xn,rowvar=False)+np.eye(d)*reg; Ct = np.cov(Tn,rowvar=False)+np.eye(d)*reg
    es,vs = np.linalg.eigh(Cs); es = np.maximum(es,1e-8)
    et,vt = np.linalg.eigh(Ct); et = np.maximum(et,1e-8)
    A = (vs@np.diag(1/np.sqrt(es))@vs.T)@(vt@np.diag(np.sqrt(et))@vt.T)
    Xn2 = Xn@A; mu = Tn.mean(0)-Xn2.mean(0)
    X[:,coral_idx] = Xn2+mu
    Xa = np.vstack([np.nan_to_num(X),np.nan_to_num(T)])
    ya = np.array([0]*len(X)+[1]*len(T))
    clf = RandomForestClassifier(150,max_depth=6,min_samples_leaf=20,random_state=42,n_jobs=-1)
    clf.fit(Xa,ya)
    p  = clf.predict_proba(np.nan_to_num(X))[:,1]
    aw = np.clip(p/(1-p+1e-6),0.05,20.0)
    rw = (years-years.min())/(years.max()-years.min()+1e-6)*0.5+0.5
    sw = aw*rw; sw /= sw.mean()
    val_mask = years>=2016
    return (X[~val_mask], y[~val_mask], sw[~val_mask],
            X[val_mask],  y[val_mask])


# ── OPTUNA OBJECTIVE ─────────────────────────────────────────────────────────
def tune(n_trials=80):
    print("Building feature matrices (takes ~3 min)...")
    X_tr, y_tr, w_tr, X_val, y_val = build_matrices()

    def objective(trial):
        p = {
            'objective':'regression','metric':'rmse','verbose':-1,
            'n_jobs':-1,'random_state':42,'n_estimators':3000,
            'num_leaves':     trial.suggest_int('num_leaves',31,255),
            'learning_rate':  trial.suggest_float('lr',0.005,0.1,log=True),
            'min_child_samples': trial.suggest_int('min_child',10,100),
            'subsample':      trial.suggest_float('subsample',0.5,1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample',0.4,1.0),
            'reg_alpha':      trial.suggest_float('alpha',1e-8,10.0,log=True),
            'reg_lambda':     trial.suggest_float('lambda',1e-8,10.0,log=True),
        }
        try:
            from optuna.integration import LightGBMPruningCallback
            cb = [lgb.early_stopping(80,verbose=False),
                  LightGBMPruningCallback(trial,'rmse')]
        except Exception:
            cb = [lgb.early_stopping(80,verbose=False)]
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr,y_tr,sample_weight=w_tr,
              eval_set=[(X_val,y_val)],callbacks=cb)
        pred = m.predict(X_val)
        return float(np.sqrt(np.mean((pred-y_val)**2)))

    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best['best_rmsle'] = study.best_value
    with open('best_params.json','w') as f:
        json.dump(best, f, indent=2)

    print(f"\n✓ Best RMSLE : {study.best_value:.4f}")
    print(f"✓ Best params: {best}")
    print("✓ Saved → best_params.json  (03_ensemble_pipeline.py will load this)")
    return best


if __name__ == '__main__':
    tune(n_trials=80)

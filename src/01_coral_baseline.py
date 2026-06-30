"""
01_coral_baseline.py
====================
CORAL Domain Adaptation Baseline for ML.ai Hackathon 2026.
Technique: Sun & Saenko (2015) — A = Cs^{-1/2} · Ct^{1/2}
Output:    submission_baseline.csv
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
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

# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────
def engineer(df, genre_s, plat_s, pub_s, yearly_s):
    df = df.copy()
    for col in ['metacritic_score','user_score','critic_review_count',
                'user_review_count','marketing_campaign_type']:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)

    pub_s2 = pub_s.rename(columns={
        'avg_estimated_revenue_million_usd':'pub_avg_rev',
        'avg_metacritic_score':'pub_avg_meta','avg_user_score':'pub_avg_user',
        'avg_launch_price_usd':'pub_avg_price','pct_sequel':'pub_pct_sequel',
        'titles':'pub_game_count'})[
        ['publisher','pub_avg_rev','pub_avg_meta','pub_avg_user',
         'pub_avg_price','pub_pct_sequel','pub_game_count']]
    df = df.merge(pub_s2, on='publisher', how='left')

    plat_s2 = plat_s.rename(columns={
        'avg_estimated_revenue_million_usd':'plat_avg_rev',
        'avg_metacritic_score':'plat_avg_meta','avg_user_score':'plat_avg_user',
        'avg_launch_price_usd':'plat_avg_price',
        'pct_online_multiplayer':'plat_pct_online','titles':'plat_game_count'})[
        ['platform','plat_avg_rev','plat_avg_meta','plat_avg_user',
         'plat_avg_price','plat_pct_online','plat_game_count']]
    df = df.merge(plat_s2, on='platform', how='left')

    genre_s2 = genre_s.rename(columns={
        'avg_estimated_revenue_million_usd':'genre_avg_rev',
        'avg_metacritic_score':'genre_avg_meta','avg_user_score':'genre_avg_user',
        'avg_launch_price_usd':'genre_avg_price',
        'pct_online_multiplayer':'genre_pct_online',
        'pct_dlc_released':'genre_pct_dlc',
        'pct_microtransactions':'genre_pct_micro','titles':'genre_game_count'})[
        ['genre','platform_type','genre_avg_rev','genre_avg_meta','genre_avg_user',
         'genre_avg_price','genre_pct_online','genre_pct_dlc',
         'genre_pct_micro','genre_game_count']]
    df = df.merge(genre_s2, on=['genre','platform_type'], how='left')

    yearly_s2 = yearly_s.rename(columns={
        'avg_estimated_revenue_million_usd':'yearly_avg_rev',
        'avg_metacritic_score':'yearly_avg_meta','avg_user_score':'yearly_avg_user',
        'avg_launch_price_usd':'yearly_avg_price','titles':'yearly_game_count'})[
        ['year','genre','yearly_avg_rev','yearly_avg_meta',
         'yearly_avg_user','yearly_avg_price','yearly_game_count']]
    df = df.merge(yearly_s2, on=['year','genre'], how='left')

    tier_map = {'Indie':0,'AA':1,'AAA':2}
    df['publisher_tier_enc'] = df['publisher_tier'].map(tier_map).fillna(-1).astype(np.int8)
    df['year_recency']  = (df['year'] - 1985) / (2026 - 1985)
    df['is_modern_era'] = (df['year'] >= 2010).astype(np.int8)
    df['is_hd_era']     = (df['year'] >= 2005).astype(np.int8)
    df['is_next_gen']   = (df['platform_generation'] >= 9).astype(np.int8)

    meta  = df['metacritic_score'].fillna(60)
    tier  = df['publisher_tier_enc'].clip(lower=0)
    price = df['launch_price_usd'].fillna(df['launch_price_usd'].median())
    df['meta_x_tier']    = meta * tier
    df['goty_x_price']   = df['goty_won'] * price
    df['dlc_x_micro']    = df['dlc_released'] * df['microtransactions']
    df['sequel_x_tier']  = df['is_sequel'] * tier
    df['online_x_micro'] = df['online_multiplayer'] * df['microtransactions']
    df['gen_x_tier']     = df['platform_generation'].fillna(0) * tier
    df['loot_x_micro']   = df['loot_boxes'] * df['microtransactions']
    df['goty_x_sequel']  = df['goty_nominated'] * df['is_sequel']

    df['completionist_ratio'] = (df['how_long_to_beat_completionist_hrs'] /
                                  (df['how_long_to_beat_main_hrs'] + 1e-3)).clip(0,20)
    df['user_critic_ratio']   = (df['user_review_count'] /
                                  (df['critic_review_count'] + 1)).clip(0,1000)
    df['score_divergence']    = df['user_score'].fillna(7)*10 - meta
    df['meta_vs_genre']       = meta - df['genre_avg_meta'].fillna(meta)
    overall_avg = df['pub_avg_rev'].mean()
    df['pub_rev_vs_avg'] = df['pub_avg_rev'].fillna(overall_avg) / (overall_avg + 1e-3)

    for col in ['publisher','developer','platform']:
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    if 'title' in df.columns:
        t = df['title'].fillna('').astype(str)
        df['title_word_count']  = t.str.split().str.len()
        df['title_is_numbered'] = t.str.contains(
            r'\s[2-9]$|\sII\b|\sIII\b|\sIV\b|\sV\b',regex=True,case=False).astype(np.int8)
        df['title_is_remaster'] = t.str.lower().str.contains(
            r'remaster|definitive|complete edition|gold edition|goty',regex=True).astype(np.int8)
    return df


def encode_cats(tr, te):
    combined = pd.concat([tr, te], ignore_index=True)
    for col in CAT_COLS:
        if col not in tr.columns: continue
        le = LabelEncoder()
        le.fit(combined[col].fillna('__MISSING__').astype(str))
        tr[col] = le.transform(tr[col].fillna('__MISSING__').astype(str))
        te[col] = le.transform(te[col].fillna('__MISSING__').astype(str))
    return tr, te


def coral_transform(X_src, X_tgt, reg=1e-5):
    d  = X_src.shape[1]
    Cs = np.cov(X_src, rowvar=False) + np.eye(d)*reg
    Ct = np.cov(X_tgt, rowvar=False) + np.eye(d)*reg
    es, vs = np.linalg.eigh(Cs); es = np.maximum(es, 1e-8)
    et, vt = np.linalg.eigh(Ct); et = np.maximum(et, 1e-8)
    A      = (vs @ np.diag(1/np.sqrt(es)) @ vs.T) @ (vt @ np.diag(np.sqrt(et)) @ vt.T)
    X_t    = X_src @ A
    mu_corr = X_tgt.mean(0) - X_t.mean(0)
    loss_b  = np.linalg.norm(Cs-Ct,'fro')**2 / (4*d**2)
    Cs_a    = np.cov(X_t+mu_corr, rowvar=False)
    loss_a  = np.linalg.norm(Cs_a-Ct,'fro')**2 / (4*d**2)
    print(f"  CORAL loss: {loss_b:.4f} → {loss_a:.8f}")
    return A, mu_corr


def adv_weights(X_tr, X_te):
    X_all = np.vstack([np.nan_to_num(X_tr), np.nan_to_num(X_te)])
    y_all = np.array([0]*len(X_tr)+[1]*len(X_te))
    clf   = RandomForestClassifier(150, max_depth=6, min_samples_leaf=20,
                                   random_state=42, n_jobs=-1)
    clf.fit(X_all, y_all)
    p = clf.predict_proba(np.nan_to_num(X_tr))[:,1]
    return np.clip(p/(1-p+1e-6), 0.05, 20.0)


# ── MAIN ────────────────────────────────────────────────────────────────────
def run():
    print("Loading data...")
    train  = pd.read_csv(f'{DATA_PATH}train_games.csv')
    test   = pd.read_csv(f'{DATA_PATH}test_features.csv')
    genre_s  = pd.read_csv(f'{DATA_PATH}genre_summary.csv')
    plat_s   = pd.read_csv(f'{DATA_PATH}platform_summary.csv')
    pub_s    = pd.read_csv(f'{DATA_PATH}publisher_summary.csv')
    yearly_s = pd.read_csv(f'{DATA_PATH}yearly_trends.csv')

    test_ids  = test['game_id'].values
    zero_mask = test['platform_type'].isin(ZERO_PLATFORMS)

    train_fe = engineer(train, genre_s, plat_s, pub_s, yearly_s)
    test_fe  = engineer(test,  genre_s, plat_s, pub_s, yearly_s)
    train_fe, test_fe = encode_cats(train_fe, test_fe)

    feat_cols = [c for c in train_fe.columns
                 if c not in set(DROP_COLS+[TARGET]) and train_fe[c].dtype!=object]
    X = train_fe[feat_cols].values.astype(float)
    T = test_fe[feat_cols].values.astype(float)
    y = np.log1p(train_fe[TARGET].values)
    years = train_fe['year'].values

    imp = SimpleImputer(strategy='median')
    X   = imp.fit_transform(X); T = imp.transform(T)

    coral_idx = [i for i,c in enumerate(feat_cols) if c in CORAL_FEATURES]
    sc = StandardScaler()
    Xn = sc.fit_transform(X[:,coral_idx]); Tn = sc.transform(T[:,coral_idx])
    print("Applying CORAL...")
    A, mu = coral_transform(Xn, Tn)
    X[:,coral_idx] = Xn @ A + mu

    print("Computing adversarial weights...")
    aw  = adv_weights(X, T)
    rw  = (years-years.min())/(years.max()-years.min()+1e-6)*0.5+0.5
    sw  = aw*rw; sw /= sw.mean()

    val_mask = years >= 2016
    params = dict(objective='regression',metric='rmse',num_leaves=127,
                  learning_rate=0.02,n_estimators=5000,min_child_samples=30,
                  subsample=0.8,subsample_freq=1,colsample_bytree=0.75,
                  reg_alpha=0.1,reg_lambda=1.0,random_state=42,n_jobs=-1,verbose=-1)

    m = lgb.LGBMRegressor(**params)
    m.fit(X[~val_mask], y[~val_mask], sample_weight=sw[~val_mask],
          eval_set=[(X[val_mask], y[val_mask])],
          callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(500)])
    val_rmsle = np.sqrt(np.mean((m.predict(X[val_mask])-y[val_mask])**2))
    print(f"Temporal RMSLE: {val_rmsle:.4f}  (best iter: {m.best_iteration_})")

    fm = lgb.LGBMRegressor(**{**params,'n_estimators':m.best_iteration_+100})
    fm.fit(X, y, sample_weight=sw)
    fm.booster_.save_model('lgbm_model.txt')

    preds = np.clip(np.expm1(fm.predict(T)), 0, None)
    preds[zero_mask.values] = 0.0

    sub = pd.DataFrame({'game_id':test_ids,
                        'estimated_revenue_million_usd':preds})
    sub.to_csv('../submission_baseline.csv', index=False)
    print(f"Saved → submission_baseline.csv  (val RMSLE={val_rmsle:.4f})")
    return fm, feat_cols, X, T, y, sw, val_rmsle

if __name__ == '__main__':
    run()

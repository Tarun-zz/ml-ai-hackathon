"""
04_shap_diagnostics.py
=======================
SHAP-based feature importance analysis.
Run AFTER 01_coral_baseline.py (needs lgbm_model.txt).
Output: assets/shap_summary.png

Install: pip install shap matplotlib
"""

import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
np.random.seed(42)

DATA_PATH      = '../'
TARGET         = 'estimated_revenue_million_usd'
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
    df = df.merge(pub_s2,on='publisher',how='left')
    plat_s2 = plat_s.rename(columns={
        'avg_estimated_revenue_million_usd':'plat_avg_rev','avg_metacritic_score':'plat_avg_meta',
        'avg_user_score':'plat_avg_user','avg_launch_price_usd':'plat_avg_price',
        'pct_online_multiplayer':'plat_pct_online','titles':'plat_game_count'})[
        ['platform','plat_avg_rev','plat_avg_meta','plat_avg_user',
         'plat_avg_price','plat_pct_online','plat_game_count']]
    df = df.merge(plat_s2,on='platform',how='left')
    genre_s2 = genre_s.rename(columns={
        'avg_estimated_revenue_million_usd':'genre_avg_rev','avg_metacritic_score':'genre_avg_meta',
        'avg_user_score':'genre_avg_user','avg_launch_price_usd':'genre_avg_price',
        'pct_online_multiplayer':'genre_pct_online','pct_dlc_released':'genre_pct_dlc',
        'pct_microtransactions':'genre_pct_micro','titles':'genre_game_count'})[
        ['genre','platform_type','genre_avg_rev','genre_avg_meta','genre_avg_user',
         'genre_avg_price','genre_pct_online','genre_pct_dlc','genre_pct_micro','genre_game_count']]
    df = df.merge(genre_s2,on=['genre','platform_type'],how='left')
    yearly_s2 = yearly_s.rename(columns={
        'avg_estimated_revenue_million_usd':'yearly_avg_rev','avg_metacritic_score':'yearly_avg_meta',
        'avg_user_score':'yearly_avg_user','avg_launch_price_usd':'yearly_avg_price',
        'titles':'yearly_game_count'})[
        ['year','genre','yearly_avg_rev','yearly_avg_meta','yearly_avg_user',
         'yearly_avg_price','yearly_game_count']]
    df = df.merge(yearly_s2,on=['year','genre'],how='left')
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


def run():
    print("Loading data...")
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

    feat_cols = [c for c in tr.columns
                 if c not in set(DROP_COLS+[TARGET]) and tr[c].dtype!=object]
    X = tr[feat_cols].values.astype(float)
    y = np.log1p(tr[TARGET].values)
    years = tr['year'].values
    val_mask = years >= 2016

    imp = SimpleImputer(strategy='median')
    X   = imp.fit_transform(X)

    T = te[feat_cols].values.astype(float)
    T = imp.transform(T)

    cidx = [i for i,c in enumerate(feat_cols) if c in CORAL_FEATURES]
    sc   = StandardScaler()
    Xn   = sc.fit_transform(X[:,cidx]); Tn = sc.transform(T[:,cidx])
    d, reg = Xn.shape[1], 1e-5
    Cs = np.cov(Xn,rowvar=False)+np.eye(d)*reg; Ct=np.cov(Tn,rowvar=False)+np.eye(d)*reg
    es,vs=np.linalg.eigh(Cs); es=np.maximum(es,1e-8)
    et,vt=np.linalg.eigh(Ct); et=np.maximum(et,1e-8)
    A=(vs@np.diag(1/np.sqrt(es))@vs.T)@(vt@np.diag(np.sqrt(et))@vt.T)
    Xn2=Xn@A; mu=Tn.mean(0)-Xn2.mean(0)
    X[:,cidx]=Xn2+mu

    # Load or train LGB
    model_path = 'lgbm_model.txt'
    if os.path.exists(model_path):
        print("Loading saved LightGBM model...")
        booster = lgb.Booster(model_file=model_path)
        model   = lgb.LGBMRegressor()
        model._Booster = booster
    else:
        print("Training LightGBM for SHAP analysis...")
        model = lgb.LGBMRegressor(objective='regression',num_leaves=127,
                                   learning_rate=0.02,n_estimators=1000,
                                   random_state=42,n_jobs=-1,verbose=-1)
        model.fit(X, y)

    # SHAP values on validation set (capped at 2000 rows for speed)
    X_shap = X[val_mask][:2000]
    print(f"Computing SHAP values on {len(X_shap)} validation samples...")
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_shap)

    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)

    print("\nTop 20 features by SHAP:")
    print(importance.head(20).round(4).to_string())
    print("\nBottom 10 features (candidates for removal):")
    print(importance.tail(10).round(6).to_string())

    # Save plot
    os.makedirs('../assets', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    top25 = importance.head(25)
    bars  = ax.barh(top25.index[::-1], top25.values[::-1], color='#534AB7')
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Feature Importance (SHAP)\nML.ai Hackathon 2026 — CORAL Pipeline',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('../assets/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved → assets/shap_summary.png")

    # Features with near-zero SHAP (potential noise)
    noise_threshold = importance.max() * 0.005
    noise_feats     = importance[importance < noise_threshold]
    if len(noise_feats):
        print(f"\nPotential noise features (SHAP < {noise_threshold:.4f}):")
        for f, v in noise_feats.items():
            print(f"  {f:40s} {v:.6f}")
        print("→ Add these to DROP_COLS in 03_ensemble_pipeline.py and retrain")


if __name__ == '__main__':
    run()

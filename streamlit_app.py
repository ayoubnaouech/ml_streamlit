import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='US Housing x Macro Dashboard (Advanced)', layout='wide')
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / 'data' / 'us_home_price_analysis_2004_2024.csv'

def find_date_col(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for i, c in enumerate(cols):
        if c in ['date','month'] or 'date' in c or 'time' in c:
            return df.columns[i]
    return None

def safe_parse_date(df: pd.DataFrame, date_col):
    if not date_col:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    return out.sort_values(date_col)

def safe_numeric(df: pd.DataFrame, date_col):
    out = df.copy()
    for c in out.columns:
        if date_col and c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def admin_label(dt):
    if pd.isna(dt):
        return 'Unknown'
    if dt < pd.Timestamp('2017-01-20'):
        return 'Pre-Trump'
    if dt <= pd.Timestamp('2021-01-19'):
        return 'Trump (2017-2020)'
    if dt <= pd.Timestamp('2025-01-19'):
        return 'Biden (2021-2024)'
    return 'Post-Biden'

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))  # sklearn-old compatible (no squared=)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def corr_matrix(df: pd.DataFrame):
    num = df.select_dtypes(include=['number'])
    if num.empty:
        return pd.DataFrame()
    return num.corr(numeric_only=True)

def build_supervised_with_lags(df: pd.DataFrame, date_col: str, target: str, features, lags=(1,3,6,12), roll_windows=(3,6,12)):
    d = df[[date_col, target] + list(features)].copy()
    d = d.sort_values(date_col).reset_index(drop=True)
    for L in lags:
        d[f'{target}_lag{L}'] = d[target].shift(L)
    for w in roll_windows:
        d[f'{target}_rollmean{w}'] = d[target].rolling(w).mean()
    d[f'{target}_diff1'] = d[target].diff(1)
    for f in features:
        d[f'{f}_lag1'] = d[f].shift(1)
    return d

def radar_compare(df: pd.DataFrame, metrics_cols):
    need = {'Trump (2017-2020)', 'Biden (2021-2024)'}
    if not need.issubset(set(df['administration'].unique())):
        return None
    d = df[list(metrics_cols) + ['administration']].copy().dropna()
    if d.empty:
        return None
    mins = d[list(metrics_cols)].min()
    maxs = d[list(metrics_cols)].max()
    scaled = (d[list(metrics_cols)] - mins) / (maxs - mins).replace(0, np.nan)
    d2 = pd.concat([scaled, d['administration']], axis=1)
    prof = d2.groupby('administration')[list(metrics_cols)].mean().loc[['Trump (2017-2020)','Biden (2021-2024)']]
    cats = list(metrics_cols)
    cats2 = cats + [cats[0]]
    trump = prof.loc['Trump (2017-2020)'].tolist()
    biden = prof.loc['Biden (2021-2024)'].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=trump+[trump[0]], theta=cats2, fill='toself', name='Trump (scaled)'))
    fig.add_trace(go.Scatterpolar(r=biden+[biden[0]], theta=cats2, fill='toself', name='Biden (scaled)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=520, title='Radar (0-1 normalized mean profile)')
    return fig

st.sidebar.title('US Housing x Macro (Advanced)')
st.sidebar.caption('Kaggle CSV + EDA + Admin comparison + ML + Forecast')
st.sidebar.markdown('**Where to put your data**:')
st.sidebar.code(str(DEFAULT_CSV), language='text')
uploaded = st.sidebar.file_uploader('Upload Kaggle CSV', type=['csv'])
use_default = st.sidebar.checkbox('Use default CSV from /data', value=(uploaded is None))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_name = f'Uploaded: {uploaded.name}'
elif use_default and DEFAULT_CSV.exists():
    df = pd.read_csv(DEFAULT_CSV)
    source_name = f'Local: {DEFAULT_CSV}'
else:
    st.error('No dataset found. Upload the CSV OR place it at data/us_home_price_analysis_2004_2024.csv')
    st.stop()

df.columns = df.columns.str.strip()
date_col = find_date_col(df)
df = safe_parse_date(df, date_col)
df = safe_numeric(df, date_col)
if date_col and df[date_col].notna().any():
    df['administration'] = df[date_col].apply(admin_label)
else:
    df['administration'] = 'Unknown'

st.sidebar.markdown('---')
show_raw = st.sidebar.checkbox('Show raw preview', value=False)
if date_col and df[date_col].notna().any():
    dmin, dmax = df[date_col].min(), df[date_col].max()
    sel = st.sidebar.slider('Date range', min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(), value=(dmin.to_pydatetime(), dmax.to_pydatetime()))
    df = df[(df[date_col] >= pd.Timestamp(sel[0])) & (df[date_col] <= pd.Timestamp(sel[1]))].copy()
if show_raw:
    st.subheader('Raw preview')
    st.dataframe(df.head(50), use_container_width=True)

st.title('US Housing & Economic Indicators — Advanced Dashboard')
st.caption(f'Source: {source_name} • Goal: explain + predict housing dynamics using macro indicators and compare patterns under Trump vs Biden.')

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric('Rows', f'{len(df):,}')
k2.metric('Columns', f'{df.shape[1]:,}')
k3.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')
k4.metric('Duplicate rows', f'{int(df.duplicated().sum()):,}')
k5.metric('Periods', f"{df['administration'].nunique():,}")
k6.metric('Date col', date_col if date_col else '—')

tabs = st.tabs(['1) Goal & Interpretation','2) Data Quality','3) Time Series','4) Correlations & Scatter','5) Charts','6) Trump vs Biden','7) ML','8) Forecast','9) Export'])

with tabs[0]:
    st.subheader('Goal & Target')
    st.markdown('- Goal: quantify how housing moves with macro indicators, compare Trump (2017-2020) vs Biden (2021-2024).')
    st.markdown('- Target: choose a housing metric (recommend Home_Price_Index if present).')
    st.markdown('- Outputs: EDA + admin comparison + ML prediction + baseline forecast + exports.')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
        target = st.selectbox('Target for interpretation', num_cols, index=num_cols.index(target_default))
        cmat = corr_matrix(df[[target] + [c for c in num_cols if c != target]])
        if not cmat.empty:
            s = cmat[target].drop(target).dropna().sort_values(key=lambda x: x.abs(), ascending=False)
            st.write('Top correlations with target (association only):')
            st.dataframe(s.head(8).reset_index().rename(columns={'index':'Feature', target:'Corr'}), use_container_width=True)

with tabs[1]:
    st.subheader('Data Quality')
    miss = df.isna().sum().sort_values(ascending=False).reset_index()
    miss.columns = ['column','missing']
    miss['missing_%'] = (miss['missing']/len(df)*100).round(2)
    st.dataframe(miss, use_container_width=True)

with tabs[2]:
    st.subheader('Time Series')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not date_col or df[date_col].isna().all():
        st.info('No usable date column detected.')
    elif not num_cols:
        st.info('No numeric columns.')
    else:
        y = st.selectbox('Metric', num_cols, index=0)
        fig = px.line(df, x=date_col, y=y, color='administration', markers=True)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        win = st.slider('Rolling window (months)', 3, 24, 12)
        tmp = df[[date_col,y]].dropna().sort_values(date_col).copy()
        tmp['rolling_mean'] = tmp[y].rolling(win).mean()
        fig2 = px.line(tmp, x=date_col, y=['rolling_mean',y])
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader('Correlations')
    cmat = corr_matrix(df)
    if cmat.empty:
        st.info('Need numeric columns.')
    else:
        fig = px.imshow(cmat, text_auto='.2f', aspect='auto', zmin=-1, zmax=1, color_continuous_scale='RdBu')
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('---')
    st.subheader('Scatter + trendline')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) >= 2:
        x = st.selectbox('X', num_cols, index=0, key='scx')
        y = st.selectbox('Y', num_cols, index=1, key='scy')
        fig = px.scatter(df, x=x, y=y, color='administration', trendline='ols', opacity=0.65)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader('Charts')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        metric = st.selectbox('Metric for distributions', num_cols, index=0)
        st.write('Box plot by administration')
        st.plotly_chart(px.box(df, x='administration', y=metric, points='outliers').update_layout(height=520), use_container_width=True)
        st.write('Violin plot by administration')
        st.plotly_chart(px.violin(df, x='administration', y=metric, box=True, points='all').update_layout(height=520), use_container_width=True)
        st.write('Radar (Spider): Trump vs Biden')
        cols = st.multiselect('Choose 3-8 metrics', num_cols, default=num_cols[:min(5,len(num_cols))])
        if cols:
            rfig = radar_compare(df, cols)
            if rfig is None:
                st.info('Need both Trump and Biden periods with enough data.')
            else:
                st.plotly_chart(rfig, use_container_width=True)

with tabs[5]:
    st.subheader('Trump vs Biden (descriptive stats)')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols = st.multiselect('Metrics', num_cols, default=num_cols[:min(6,len(num_cols))], key='adm_metrics')
    if cols:
        grp = df.groupby('administration')[cols].agg(['mean','median','std','min','max']).round(3)
        st.dataframe(grp, use_container_width=True)
        if {'Trump (2017-2020)','Biden (2021-2024)'} <= set(df['administration'].unique()):
            gmean = df.groupby('administration')[cols].mean(numeric_only=True)
            delta = (gmean.loc['Biden (2021-2024)'] - gmean.loc['Trump (2017-2020)']).sort_values()
            st.plotly_chart(px.bar(delta, orientation='h', labels={'value':'Delta (Biden - Trump)','index':'Metric'}).update_layout(height=450), use_container_width=True)

with tabs[6]:
    st.subheader('ML Prediction')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        st.stop()
    target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
    target = st.selectbox('Target', num_cols, index=num_cols.index(target_default), key='ml_target')
    feature_candidates = [c for c in num_cols if c != target]
    feats = st.multiselect('Features', feature_candidates, default=feature_candidates[:min(10,len(feature_candidates))])
    if not feats:
        st.info('Select at least one feature.')
        st.stop()
    use_scaling = st.checkbox('Normalize features', value=True)
    scaler_type = st.selectbox('Scaling method', ['StandardScaler (z-score)','MinMaxScaler (0-1)'])
    scaler = StandardScaler() if scaler_type.startswith('Standard') else MinMaxScaler()
    model_choice = st.selectbox('Model', ['Ridge','RandomForest'], index=1)
    d = df[[target] + feats].copy().dropna(subset=[target])
    split = int(len(d)*0.8)
    X_train, X_test = d[feats].iloc[:split], d[feats].iloc[split:]
    y_train, y_test = d[target].iloc[:split], d[target].iloc[split:]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaling:
        steps.append(('scaler', scaler))
    if model_choice == 'Ridge':
        steps.append(('model', Ridge(alpha=1.0, random_state=0)))
    else:
        steps.append(('model', RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1, min_samples_leaf=2)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae, rmse, r2 = metrics(y_test, pred)
    c1,c2,c3 = st.columns(3)
    c1.metric('MAE', f'{mae:.3f}')
    c2.metric('RMSE', f'{rmse:.3f}')
    c3.metric('R2', f'{r2:.3f}')
    out = pd.DataFrame({'actual': y_test.values, 'pred': pred})
    st.plotly_chart(px.line(out, y=['actual','pred']).update_layout(height=420), use_container_width=True)
    st.subheader('Drivers')
    if model_choice == 'RandomForest':
        imp = pd.Series(model.named_steps['model'].feature_importances_, index=feats).sort_values()
        st.plotly_chart(px.bar(imp.tail(20), orientation='h', labels={'value':'Importance','index':'Feature'}).update_layout(height=520), use_container_width=True)
    else:
        coefs = pd.Series(model.named_steps['model'].coef_, index=feats).sort_values()
        st.plotly_chart(px.bar(coefs, orientation='h', labels={'value':'Coefficient','index':'Feature'}).update_layout(height=520), use_container_width=True)

with tabs[7]:
    st.subheader('Forecast (Future results)')
    st.caption('Baseline forecast: uses target lags; macro held constant at last observed value.')
    if not date_col or df[date_col].isna().all():
        st.stop()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
    target = st.selectbox('Target to forecast', num_cols, index=num_cols.index(target_default), key='fc_target')
    exog = st.multiselect('Exogenous features (held constant)', [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:min(6,len(num_cols)-1)])
    horizon = st.slider('Horizon (months)', 6, 24, 12)
    model_choice = st.selectbox('Forecast model', ['Ridge','RandomForest'], index=0)
    use_scaling = st.checkbox('Normalize (forecast)', value=True, key='fc_norm')
    scaler_type = st.selectbox('Scaler (forecast)', ['StandardScaler (z-score)','MinMaxScaler (0-1)'], key='fc_scaler')
    scaler = StandardScaler() if scaler_type.startswith('Standard') else MinMaxScaler()
    d0 = df[[date_col, target] + exog].copy().sort_values(date_col).reset_index(drop=True)
    for f in exog:
        d0[f] = d0[f].ffill()
    sup = build_supervised_with_lags(d0, date_col, target, exog).dropna(subset=[target])
    feature_cols = [c for c in sup.columns if c not in [date_col, target]]
    split = max(10, len(sup) - horizon)
    train = sup.iloc[:split].copy()
    X_train, y_train = train[feature_cols], train[target]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaling:
        steps.append(('scaler', scaler))
    if model_choice == 'Ridge':
        steps.append(('model', Ridge(alpha=1.0, random_state=0)))
    else:
        steps.append(('model', RandomForestRegressor(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    last_date = d0[date_col].dropna().max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
    tmp = d0.copy()
    preds = []
    for dt in future_dates:
        new_row = {date_col: dt, target: np.nan}
        for f in exog:
            new_row[f] = tmp[f].iloc[-1] if len(tmp) else np.nan
        tmp = pd.concat([tmp, pd.DataFrame([new_row])], ignore_index=True)
        sup_tmp = build_supervised_with_lags(tmp, date_col, target, exog)
        last_feat = sup_tmp.iloc[[-1]][feature_cols]
        yhat = float(model.predict(last_feat)[0])
        tmp.loc[tmp.index[-1], target] = yhat
        preds.append(yhat)
    forecast_df = pd.DataFrame({date_col: future_dates, 'forecast': preds})
    hist = d0[[date_col, target]].dropna().copy()
    fig = px.line(hist, x=date_col, y=target, title='History + Forecast')
    fig.add_scatter(x=forecast_df[date_col], y=forecast_df['forecast'], mode='lines+markers', name='Forecast')
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)
    st.download_button('Download forecast CSV', data=forecast_df.to_csv(index=False).encode('utf-8'), file_name='forecast.csv', mime='text/csv')

with tabs[8]:
    st.subheader('Export')
    st.download_button('Download filtered dataset CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='filtered_us_housing.csv', mime='text/csv')

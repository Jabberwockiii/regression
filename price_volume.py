import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, kpss, coint

# Configure page
st.set_page_config(page_title="Bitcoin Analysis", layout="wide", page_icon="₿")
st.title("💰 Bitcoin Price-Volume Relationship Analyzer")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2017-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
degree = st.sidebar.slider("Polynomial Degree", 1, 3, 1)
log_transform = st.sidebar.checkbox("Log Transform", True)
show_raw = st.sidebar.checkbox("Show Raw Data", False)

@st.cache_data
def load_data(start_date, end_date):
    """Load Bitcoin data from Yahoo Finance"""
    btc = yf.Ticker("BTC-USD")
    df = btc.history(start=start_date, end=end_date)[['Close', 'Volume']]
    df.columns = ['price', 'volume']
    return df.dropna()

def prepare_data(df, log_transform):
    """Prepare data for analysis"""
    if log_transform:
        df['log_price'] = np.log(df['price'])
        df['log_volume'] = np.log(df['volume'])
    return df

def run_regression(df, degree=1, log_transform=False):
    """Perform regression analysis"""
    if log_transform:
        X = df[['log_volume']].values
        y = df['log_price'].values
        x_label = "Log Volume"
        y_label = "Log Price"
    else:
        X = df[['volume']].values
        y = df['price'].values
        x_label = "Volume (USD)"
        y_label = "Price (USD)"
    
    # Polynomial features if degree > 1
    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        X = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Create prediction line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    if degree > 1:
        x_range_poly = poly.transform(x_range)
        y_pred = model.predict(x_range_poly)
    else:
        y_pred = model.predict(x_range)
    
    # Calculate metrics
    y_pred_full = model.predict(X)
    r2 = model.score(X, y)
    if degree == 1:
        slope = model.coef_[0] if degree == 1 else None
        intercept = model.intercept_
    else:
        slope = None
        intercept = None
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(X[:,0], y) if degree == 1 else (None, None)
    
    return {
        'x_range': x_range,
        'y_pred': y_pred,
        'r2': r2,
        'slope': slope,
        'intercept': intercept,
        'corr': corr,
        'p_value': p_value,
        'x_label': x_label,
        'y_label': y_label
    }

def perform_stationarity_tests(df, log_transform):
    """Perform and display stationarity tests"""
    if log_transform:
        series_to_test = {
            'Log Price': df['log_price'],
            'Log Volume': df['log_volume']
        }
    else:
        series_to_test = {
            'Price': df['price'],
            'Volume': df['volume']
        }
    
    results = []
    for name, series in series_to_test.items():
        # ADF Test
        adf_result = adfuller(series.dropna())
        # KPSS Test
        kpss_result = kpss(series.dropna())
        
        results.append({
            'Series': name,
            'ADF Statistic': adf_result[0],
            'ADF p-value': adf_result[1],
            'ADF Critical Values': str(adf_result[4]),
            'ADF Conclusion': "Stationary" if adf_result[1] < 0.05 else "Non-Stationary",
            'KPSS Statistic': kpss_result[0],
            'KPSS p-value': kpss_result[1],
            'KPSS Critical Values': str(kpss_result[3]),
            'KPSS Conclusion': "Stationary" if kpss_result[1] > 0.05 else "Non-Stationary"
        })
    
    return pd.DataFrame(results)

# Main analysis
df = load_data(start_date, end_date)
df = prepare_data(df, log_transform)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "📊 Statistics", "🔍 Differenced Analysis", "ℹ️ About"])

with tab1:
    st.header("Price-Volume Relationship")
    
    # Run regression
    results = run_regression(df, degree, log_transform)
    
    # Create interactive plot
    if log_transform:
        fig = px.scatter(df, x='log_volume', y='log_price', 
                         hover_data={'price': ':.2f', 'volume': ':.2f'},
                         labels={'log_volume': 'Log Trading Volume', 
                                'log_price': 'Log Price'})
    else:
        fig = px.scatter(df, x='volume', y='price',
                         hover_data={'price': ':.2f', 'volume': ':.2f'})
    
    fig.add_traces(px.line(x=results['x_range'].flatten(), 
                       y=results['y_pred']).data[0])
    fig.update_traces(line_color='red', name='Regression Line')
    fig.update_layout(showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R-squared", f"{results['r2']:.4f}")
    with col2:
        if degree == 1:
            st.metric("Correlation", f"{results['corr']:.4f}")
    with col3:
        if degree == 1:
            st.metric("P-value", f"{results['p_value']:.4f}")

with tab2:
    st.header("Statistical Summary")
    
    if log_transform:
        st.write("Log-Transformed Data Statistics:")
        st.dataframe(df[['log_price', 'log_volume']].describe().style.format("{:.4f}"))
    else:
        st.write("Raw Data Statistics:")
        st.dataframe(df[['price', 'volume']].describe().style.format("{:.2f}"))
    
    # Stationarity Tests
    st.subheader("Stationarity Tests")
    stationarity_results = perform_stationarity_tests(df, log_transform)
    
    # Display ADF results
    st.write("### Augmented Dickey-Fuller (ADF) Test")
    adf_display = stationarity_results[['Series', 'ADF Statistic', 'ADF p-value', 'ADF Conclusion']]
    st.dataframe(adf_display.style.format({
        'ADF Statistic': '{:.4f}',
        'ADF p-value': '{:.4f}'
    }))
    
    # Display KPSS results
    st.write("### KPSS Test")
    kpss_display = stationarity_results[['Series', 'KPSS Statistic', 'KPSS p-value', 'KPSS Conclusion']]
    st.dataframe(kpss_display.style.format({
        'KPSS Statistic': '{:.4f}',
        'KPSS p-value': '{:.4f}'
    }))
    
    # Cointegration test
    st.subheader("Cointegration Test")
    if log_transform:
        coint_result = coint(df['log_price'], df['log_volume'])
    else:
        coint_result = coint(df['price'], df['volume'])
    
    st.write(f"Cointegration test p-value: {coint_result[1]:.4f}")
    if coint_result[1] < 0.05:
        st.success("✅ Significant cointegration (long-term equilibrium exists)")
    else:
        st.warning("⚠️ No significant cointegration found")
    
    if show_raw:
        st.write("Raw Data Preview:")
        st.dataframe(df.head())

with tab3:
    st.header("Differenced Series Analysis")
    
    # Select difference order
    diff_order = st.selectbox("Select difference order", [1, 2], key="diff_order_analysis")
    
    # Prepare differenced data
    df_diff = df.copy()
    if log_transform:
        df_diff[f'diff{diff_order}_log_price'] = df_diff['log_price'].diff(diff_order)
        df_diff[f'diff{diff_order}_log_volume'] = df_diff['log_volume'].diff(diff_order)
        x_col = f'diff{diff_order}_log_volume'
        y_col = f'diff{diff_order}_log_price'
        x_label = f"Diff {diff_order} Log Volume"
        y_label = f"Diff {diff_order} Log Price"
    else:
        df_diff[f'diff{diff_order}_price'] = df_diff['price'].diff(diff_order)
        df_diff[f'diff{diff_order}_volume'] = df_diff['volume'].diff(diff_order)
        x_col = f'diff{diff_order}_volume'
        y_col = f'diff{diff_order}_price'
        x_label = f"Diff {diff_order} Volume"
        y_label = f"Diff {diff_order} Price"
    
    # Remove NaN values
    df_diff = df_diff.dropna(subset=[x_col, y_col])
    
    # Run regression
    X = df_diff[[x_col]].values
    y = df_diff[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Create prediction line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    # Calculate metrics
    r2 = model.score(X, y)
    corr, p_value = stats.pearsonr(X.flatten(), y)
    
    # Plot regression
    fig = px.scatter(df_diff, x=x_col, y=y_col,
                   labels={x_col: x_label, y_col: y_label},
                   trendline="ols",
                   title=f"{diff_order}-order Differenced Series Regression",
                   hover_data={'date': df_diff.index.strftime('%Y-%m-%d')})
    
    # Add custom regression line
    fig.add_traces(px.line(x=x_range.flatten(), y=y_pred, 
                         color_discrete_sequence=['red']).data[0])
    fig.update_traces(line_dash="dash", name='Regression Line')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R-squared", f"{r2:.4f}")
    with col2:
        st.metric("Correlation", f"{corr:.4f}")
    with col3:
        st.metric("P-value", f"{p_value:.4f}")
    
    # Show differenced data stats
    st.subheader("Differenced Data Statistics")
    st.dataframe(df_diff[[x_col, y_col]].describe().style.format("{:.4f}"))
    
    # Show stationarity tests for differenced series
    st.subheader("Stationarity Tests for Differenced Series")
    if log_transform:
        adf_price = adfuller(df_diff[f'diff{diff_order}_log_price'].dropna())
        adf_volume = adfuller(df_diff[f'diff{diff_order}_log_volume'].dropna())
    else:
        adf_price = adfuller(df_diff[f'diff{diff_order}_price'].dropna())
        adf_volume = adfuller(df_diff[f'diff{diff_order}_volume'].dropna())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"#### {y_label} - ADF Test")
        st.write(f"Statistic: {adf_price[0]:.4f}")
        st.write(f"p-value: {adf_price[1]:.4f}")
        st.write("Critical Values:", adf_price[4])
        st.write("**Conclusion:**", "Stationary" if adf_price[1] < 0.05 else "Non-Stationary")
    
    with col2:
        st.write(f"#### {x_label} - ADF Test")
        st.write(f"Statistic: {adf_volume[0]:.4f}")
        st.write(f"p-value: {adf_volume[1]:.4f}")
        st.write("Critical Values:", adf_volume[4])
        st.write("**Conclusion:**", "Stationary" if adf_volume[1] < 0.05 else "Non-Stationary")
    
    # Show raw differenced values
    if st.checkbox("Show raw differenced values", False, key="show_diff_raw"):
        st.dataframe(df_diff[[x_col, y_col]].head())

with tab4:
    st.header("About This Analysis")
    st.markdown("""
    ### Bitcoin Price-Volume Relationship Analyzer
    This interactive tool explores the relationship between Bitcoin's price and trading volume.
    
    **Features:**
    - Adjustable date range
    - Log transformation option
    - Polynomial regression (1st-3rd degree)
    - Interactive visualizations
    - Statistical metrics
    - Stationarity testing (ADF and KPSS tests)
    - Differenced series analysis
    - Cointegration testing
    
    **How to Use:**
    1. Adjust parameters in the sidebar
    2. View results in the analysis tabs:
       - Main regression analysis
       - Statistical tests
       - Differenced series analysis
    3. Check for long-term equilibrium (cointegration)
    
    **Technical Details:**
    - Data from Yahoo Finance
    - Linear regression with scikit-learn
    - Stationarity tests with statsmodels
    - Visualization with Plotly
    - Built with Streamlit
    """)

# Add some style
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.2rem;
}
.css-1aumxhk {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)
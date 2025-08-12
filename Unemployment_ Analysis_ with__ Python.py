# Step 0: imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer
from scipy import stats

sns.set(style='whitegrid', context='talk')
plt.rcParams['figure.figsize'] = (12,6)

# ---------- Step 1: Data acquisition ----------
INPUT_CSV = 'unemployment.csv'  # change if needed

if os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}")
else:
    print(f"{INPUT_CSV} not found. Generating example monthly series (2008-01 to 2024-12). Replace with your CSV.")
    rng = pd.date_range('2008-01-01', '2024-12-01', freq='MS')
    np.random.seed(42)
    base = 5 + 0.01 * np.arange(len(rng))
    seasonal = 0.8 * np.sin(2 * np.pi * rng.month / 12)
    # Removed COVID-related calculation from here to avoid NameError when loading CSV
    noise = np.random.normal(0, 0.4, len(rng))
    series = base + seasonal + noise # Removed covid from series
    df = pd.DataFrame({'date': rng, 'unemployment_rate': series})

# ---------- Step 2: Parse dates and set index ----------
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')
df = df.asfreq('MS')  # monthly frequency

# ---------- Step 3: Missing values handling ----------
missing_pct = df['unemployment_rate'].isna().mean() * 100
print(f"Missing values: {missing_pct:.2f}%")

df['unemployment_rate'] = df['unemployment_rate'].interpolate(method='time')
df['unemployment_rate'] = df['unemployment_rate'].fillna(method='ffill').fillna(method='bfill')

df.to_csv('unemployment_cleaned.csv')

# ---------- Step 4: Quick EDA ----------
print("\nSummary statistics:")
print(df['unemployment_rate'].describe())

plt.figure()
plt.plot(df.index, df['unemployment_rate'], label='Unemployment rate', linewidth=1.5)
plt.plot(df['unemployment_rate'].rolling(window=12).mean(), label='12-month rolling mean', linewidth=2)
plt.title('Unemployment Rate Time Series')
plt.ylabel('Unemployment rate (%)')
plt.legend()
plt.tight_layout()
plt.savefig('ts_unemployment.png')
plt.show()

df['yoy_abs'] = df['unemployment_rate'].diff(12)
df['yoy_pct'] = df['unemployment_rate'].pct_change(12) * 100

plt.figure()
plt.plot(df.index, df['yoy_abs'])
plt.axhline(0, color='black', lw=0.8)
plt.title('Year-on-year absolute change in unemployment rate')
plt.ylabel('Percentage points')
plt.tight_layout()
plt.savefig('yoy_abs.png')
plt.show()

# ---------- Step 5: Seasonal decomposition (STL) ----------
stl = STL(df['unemployment_rate'], period=12, robust=True)
res = stl.fit()
fig = res.plot()
fig.suptitle('STL Decomposition (trend, seasonal, resid)')
plt.tight_layout()
plt.savefig('stl_decomp.png')
plt.show()

# ---------- Step 6: ACF/PACF ----------
plot_acf(df['unemployment_rate'].dropna(), lags=36)
plt.title('Autocorrelation (ACF)')
plt.savefig('acf.png')
plt.show()

plot_pacf(df['unemployment_rate'].dropna(), lags=36, method='ywm')
plt.title('Partial Autocorrelation (PACF)')
plt.savefig('pacf.png')
plt.show()

# ---------- Step 7: COVID impact analysis ----------
# Moved COVID analysis after data loading/generation to ensure covid_start is always defined
covid_start = pd.to_datetime('2020-03-01')
pre_period = df[df.index < covid_start]
post_period = df[df.index >= covid_start]

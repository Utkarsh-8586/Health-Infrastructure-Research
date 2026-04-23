import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("Rural Health Statistics Districtwise Health Care Infrastructure.csv")
print(df.head())

# ── Initial Inspection ────────────────────────────────────────────────────────
print(df.info())
print(df.isnull().sum())
print(df.nunique())
df.describe()

# ── Rename Columns ────────────────────────────────────────────────────────────
df.columns = df.columns.str.strip()

df = df.rename(columns={
    'Functional Sub Centres (UOM:Number), Scaling Factor:1': 'Functional Sub Centres',
    'Functional Primary Health Centres (Phcs) (UOM:Number), Scaling Factor:1': 'Functional Primary Health Centres',
    'Functional Community Health Centres (Chcs) (UOM:Number), Scaling Factor:1': 'Functional Community Health Centres',
    'Functional Health And Wellness Centres-Sub Centres  (Hwc-Scs) (UOM:Number), Scaling Factor:1': 'HWC_SubCentres',
    'Functional Health And Wellness Centres (Hwc)-Primary Health Centres (Phcs) (UOM:Number), Scaling Factor:1': 'HWC_PHCs',
    'Sub_Divisional_Hospitals': 'Sub_Div_Hospitals',
    'District_Hospitals': 'District_Hospitals',
    'Functional Sub Divisional Hospitals (Sdhs) (UOM:Number), Scaling Factor:1': 'Functional Sub Divisional Hospitals',
    'Functional District Hospitals (Dhs) (UOM:Number), Scaling Factor:1': 'Functional District Hospitals'
})

threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=threshold)

# ── Clean Data ────────────────────────────────────────────────────────────────
df.drop_duplicates()
df = df.dropna(how='all')

df['State'] = df['State'].astype(str).str.strip()
df['District'] = df['District'].astype(str).str.strip()

df = df[~df['District'].str.contains('Unknown Districts Of India', case=False, na=False)]
invalid_names = ['Central', 'North', 'South', 'East', 'West', 'Unknown']
df = df[~df['District'].isin(invalid_names)]

df['Year'] = df['Year'].str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

cols_to_convert = [
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres',
    'Functional Sub Divisional Hospitals',
    'Functional District Hospitals'
]

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[cols_to_convert] = df[cols_to_convert].fillna(0)
df[cols_to_convert] = df[cols_to_convert].astype(int)

df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)
df = df.reset_index(drop=True)

print(df.dtypes)
print(df.info())
print(df.head())

# ── Plot: Top 5 Districts by PHCs ─────────────────────────────────────────────
latest_year = df['Year'].max()
df_latest = df[df['Year'] == latest_year]

top_phc = df_latest.sort_values(by='Functional Primary Health Centres', ascending=False).head(5)

colors = ['#6A5ACD', '#FF7F50', '#20B2AA', '#3CB371', '#9ACD32']
plt.figure(figsize=(10, 6))
plt.barh(top_phc['District'], top_phc['Functional Primary Health Centres'], color=colors, edgecolor='black')
plt.title(f"Top 5 Districts by PHCs in {latest_year}", fontsize=13, fontweight='semibold')
plt.xlabel("PHC Count")
plt.ylabel("District Name")
plt.gca().invert_yaxis()
for i, v in enumerate(top_phc['Functional Primary Health Centres']):
    plt.text(v + 0.3, i, f'{v}', fontsize=9)
plt.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# ── Plot: Top 5 Districts by CHCs ─────────────────────────────────────────────
top_chc = df_latest.sort_values(by='Functional Community Health Centres', ascending=False).head(5)

colors = ['#5B8FF9', '#FF9D4D', '#5AD8A6', '#5D7092', '#B37FEB']
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.barh(top_chc['District'], top_chc['Functional Community Health Centres'], color=colors, edgecolor='black')
plt.title(f"Top 5 Districts by CHCs - {latest_year}", fontsize=13, fontweight='semibold')
plt.xlabel("CHC Count")
plt.ylabel("District Name")
plt.gca().invert_yaxis()
for i, v in enumerate(top_chc['Functional Community Health Centres']):
    plt.text(v + 0.2, i, f'{v}', va='center', fontsize=9)
plt.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

# ── Plot: Top 10 States – Stacked Bar ─────────────────────────────────────────
state_data = df_latest.groupby('State')[[
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres'
]].sum().sort_values(by='Functional Sub Centres', ascending=False).head(10)

plt.figure(figsize=(12, 6))
colors = ['#7B68EE', '#FFA500', '#2E8B57']
state_data.plot(kind='bar', stacked=True, color=colors, edgecolor='black')
plt.title(f"Top 10 States - Stacked Healthcare Facilities ({latest_year})", fontsize=13, fontweight='semibold')
plt.xlabel("States")
plt.ylabel("Total Facilities")
plt.xticks(rotation=40, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title="Facility Type")
plt.tight_layout()
plt.show()

# ── Plot: Year-wise Trend ──────────────────────────────────────────────────────
yearly = df.groupby('Year')[[
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres'
]].mean()

plt.figure(figsize=(10, 6))
plt.plot(yearly.index, yearly['Functional Sub Centres'], marker='o', linewidth=2)
plt.plot(yearly.index, yearly['Functional Primary Health Centres'], marker='s', linewidth=2)
plt.plot(yearly.index, yearly['Functional Community Health Centres'], marker='^', linewidth=2)
plt.title("Year-wise Trend of Health Infrastructure", fontsize=13, fontweight='semibold')
plt.xlabel("Year")
plt.ylabel("Average Count")
plt.grid(linestyle='--', alpha=0.4)
plt.legend(['Sub Centres', 'Primary Health Centres', 'Community Health Centres'])
plt.tight_layout()
plt.show()

# ── Plot: Top 5 Districts by Sub Centres ──────────────────────────────────────
top_sc = df_latest.sort_values(by='Functional Sub Centres', ascending=False).head(5)

colors = ['#6A5ACD', '#FF7F50', '#20B2AA', '#3CB371', '#9ACD32']
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.barh(top_sc['District'], top_sc['Functional Sub Centres'], color=colors, edgecolor='black')
plt.title(f"Top 5 Districts by Sub Centres - {int(latest_year)}", fontsize=13, fontweight='semibold')
plt.xlabel('Sub Centre Count')
plt.ylabel('District Name')
plt.gca().invert_yaxis()
for i, v in enumerate(top_sc['Functional Sub Centres']):
    plt.text(v + 0.5, i, f'{v}', va='center', fontsize=9)
plt.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

# ── Plot: Pie Chart – Facility Distribution ───────────────────────────────────
total = df_latest[[
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres'
]].sum()

explode = (0.08, 0.02, 0.02)
colors = ['#7B68EE', '#FFA500', '#2E8B57']
plt.figure(figsize=(7, 7))
plt.pie(total, labels=total.index, autopct='%1.1f%%', explode=explode, colors=colors, shadow=True, startangle=140)
plt.title(f"Healthcare Facility Distribution - {int(latest_year)}", fontsize=13, fontweight='semibold')
plt.tight_layout()
plt.show()

# ── Plot: Box Plot ────────────────────────────────────────────────────────────
data = df_latest[[
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres'
]]

colors = ['#6A5ACD', '#FF7F50', '#3CB371']
plt.figure(figsize=(8, 6))
box = plt.boxplot(data.values, patch_artist=True, widths=0.5)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in box['medians']:
    median.set(color='black', linewidth=1.5)
plt.xticks([1, 2, 3], ['Sub Centres', 'PHCs', 'CHCs'])
plt.ylabel("Facility Count")
plt.title("Healthcare Facilities Distribution", fontsize=13, fontweight='semibold')
plt.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

# ── Plot: Scatter – Sub Centres vs PHCs ───────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(df_latest['Functional Sub Centres'], df_latest['Functional Primary Health Centres'],
            color='#6A5ACD', alpha=0.6, edgecolors='black')
plt.xlabel("Sub Centres")
plt.ylabel("PHCs")
plt.title("Relationship Between Sub Centres and PHCs", fontsize=13, fontweight='semibold')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

# ── Plot: Correlation Heatmap ─────────────────────────────────────────────────
corr = df_latest[[
    'Functional Sub Centres',
    'Functional Primary Health Centres',
    'Functional Community Health Centres',
    'Functional District Hospitals'
]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation of Healthcare Facilities", fontsize=13, fontweight='semibold')
plt.tight_layout()
plt.show()

# ── ML Model: Linear Regression ───────────────────────────────────────────────
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

features_num = ['Functional Sub Centres', 'Functional Primary Health Centres', 'Year']
features_cat = ['State']
target = 'Functional Community Health Centres'

df_model = df[features_num + features_cat + [target]].dropna()
X = df_model[features_num + features_cat]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

preprocessor = ColumnTransformer([
    ('num', 'passthrough', features_num),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f'MAE : {mae:.2f}')
print(f'R2  : {r2:.4f}')

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='#6A5ACD', edgecolors='black', linewidths=0.4)
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray')
plt.xlabel('Actual CHCs')
plt.ylabel('Predicted CHCs')
plt.title(f'Actual vs Predicted CHCs\nMAE={mae:.2f} | R2={r2:.4f}', fontsize=12, fontweight='semibold')
plt.tight_layout()
plt.show()

# Feature Coefficients
lr = model.named_steps['regressor']
num_coefs = lr.coef_[:len(features_num)]
colors = ['#6A5ACD', '#FF7F50', '#3CB371']
plt.figure(figsize=(6, 4))
plt.barh(features_num, num_coefs, color=colors, edgecolor='black')
plt.xlabel('Coefficient')
plt.title('Feature Coefficients (Numeric)', fontsize=12, fontweight='semibold')
plt.tight_layout()
plt.show()

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

with st.sidebar:
    st.markdown("Hilmy")
    st.caption("Market Basket Analysis")

df = pd.read_csv("bread_basket.csv")
df.columns = df.columns.str.strip().str.lower()

df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
df['month'] = df['date_time'].dt.month_name()
df['day'] = df['date_time'].dt.day_name()

df['hour'] = df['date_time'].dt.hour
df['period_day'] = pd.cut(
    df['hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False
)

df['weekday_weekend'] = df['day'].isin(['Saturday','Sunday']).map({
    True: 'Weekend',
    False: 'Weekday'
})

def get_data(period_day, weekday_weekend, month, day):
    filtered = df[
        (df['period_day'] == period_day.title()) &
        (df['weekday_weekend'] == weekday_weekend.title()) &
        (df['month'].str.startswith(month)) &
        (df['day'].str.startswith(day))
    ]
    return filtered if not filtered.empty else "No Result!"

def user_input_features():
    item = st.selectbox("Item", sorted(df['item'].unique()))
    period_day = st.selectbox('Period day', ['Morning','Afternoon','Evening','Night'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekday','Weekend'])
    month = st.select_slider('Month', ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    day = st.select_slider('Day', ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], value="Sat")
    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_features()
data = get_data(period_day, weekday_weekend, month, day)

def encode(x):
    return 1 if x > 0 else 0

if isinstance(data, pd.DataFrame):
    basket = (
        data.groupby(['transaction', 'item'])['item']
        .count().unstack().fillna(0)
        .applymap(encode)
    )

    frequent_items = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
def recommend(item):
    rec = rules[rules['antecedents'].str.contains(item)]
    return rec.iloc[0]['consequents'] if not rec.empty else None

if isinstance(data, pd.DataFrame):
    rekomendasi = recommend(item)
    if rekomendasi:
        st.success(f"Jika membeli **{item}**, biasanya juga membeli **{rekomendasi}**")
    else:
        st.warning("Tidak ada rekomendasi.")

import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google_play_scraper import Sort, reviews

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

MASTER_CSV = 'nvidia_daily_master_360.csv'
REVIEWS_CSV = 'geforcenow_app_reviews_raw.csv'

print("--- 1. Loading existing datasets ---")
df_master = pd.read_csv(MASTER_CSV)
df_reviews = pd.read_csv(REVIEWS_CSV)

df_master['date'] = pd.to_datetime(df_master['date'])
last_master_date = df_master['date'].max().date()
yesterday = datetime.date.today() - datetime.timedelta(days=1)

print(f"Master latest date: {last_master_date} | Target date: {yesterday}")

print("--- 2. Scraping recent Google Play reviews for GeForce NOW ---")
new_reviews_raw, _ = reviews(
    'com.nvidia.geforcenow',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=250
)

if new_reviews_raw:
    df_new_rev = pd.DataFrame(new_reviews_raw)
    df_new_rev = df_new_rev[['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at']]
    df_new_rev.rename(columns={
        'reviewId': 'review_id',
        'userName': 'author_name',
        'content': 'review_text',
        'score': 'rating',
        'thumbsUpCount': 'thumbs_up',
        'at': 'posted_at'
    }, inplace=True)

    existing_ids = set(df_reviews['review_id'].astype(str))
    df_new_rev = df_new_rev[~df_new_rev['review_id'].astype(str).isin(existing_ids)].copy()

    if not df_new_rev.empty:
        print(f"Adding {len(df_new_rev)} new GeForce NOW user reviews...")
        df_new_rev['posted_at'] = pd.to_datetime(df_new_rev['posted_at']).dt.tz_localize(None)
        df_new_rev['sentiment_score'] = df_new_rev['review_text'].astype(str).apply(
            lambda t: sia.polarity_scores(t)['compound']
        )
        df_new_rev['sentiment_category'] = pd.cut(
            df_new_rev['sentiment_score'],
            bins=[-1.01, -0.05, 0.05, 1.01],
            labels=['Negative', 'Neutral', 'Positive']
        )
        df_reviews = pd.concat([df_new_rev, df_reviews], ignore_index=True)
        df_reviews.to_csv(REVIEWS_CSV, index=False)
        print(f"Updated '{REVIEWS_CSV}' successfully.")
    else:
        print("No new unique reviews found.")
else:
    print("Scraper returned no reviews.")

print("--- 3. Updating daily master time series ---")
if last_master_date >= yesterday:
    print("Daily master timeline is already up to date.")
else:
    fetch_start = (last_master_date - datetime.timedelta(days=250)).strftime("%Y-%m-%d")
    fetch_end = (datetime.date.today()).strftime("%Y-%m-%d")

    tickers = {
        'NVDA': 'nvda',
        'TSM': 'tsm',
        'SMH': 'smh_etf',
        'QQQ': 'qqq'
    }

    raw_market = yf.download(list(tickers.keys()), start=fetch_start, end=fetch_end, group_by='ticker', auto_adjust=False)

    market_dfs = []
    for ticker, prefix in tickers.items():
        sub = raw_market[ticker][['Close', 'Volume']].copy() if ticker == 'NVDA' else raw_market[ticker][['Close']].copy()
        sub.columns = [f"{prefix}_{col.lower()}" for col in sub.columns]
        market_dfs.append(sub)

    mkt = pd.concat(market_dfs, axis=1)
    mkt.index = pd.to_datetime(mkt.index).tz_localize(None)

    mkt['nvda_return_pct'] = mkt['nvda_close'].pct_change() * 100
    mkt['nvda_volatility_30d'] = mkt['nvda_return_pct'].rolling(window=30).std()
    mkt['nvda_sma_50'] = mkt['nvda_close'].rolling(window=50).mean()
    mkt['nvda_sma_200'] = mkt['nvda_close'].rolling(window=200).mean()

    # Aggregate sentiment daily
    df_reviews['date_str'] = pd.to_datetime(df_reviews['posted_at']).dt.strftime('%Y-%m-%d')
    daily_sent = df_reviews.groupby('date_str').agg(
        gfn_reviews_count=('review_id', 'count'),
        gfn_avg_rating=('rating', 'mean'),
        gfn_sentiment_score=('sentiment_score', 'mean')
    ).reset_index().rename(columns={'date_str': 'date'})
    daily_sent['date'] = pd.to_datetime(daily_sent['date'])

    new_dates = pd.date_range(start=last_master_date + datetime.timedelta(days=1), end=yesterday, freq='D')
    new_rows = pd.DataFrame({'date': new_dates})

    new_rows = new_rows.merge(mkt.reset_index().rename(columns={'Date': 'date'}), on='date', how='left')
    new_rows['is_trading_day'] = new_rows['nvda_close'].notnull().astype(int)

    new_rows = new_rows.merge(daily_sent, on='date', how='left')
    new_rows['gfn_reviews_count'] = new_rows['gfn_reviews_count'].fillna(0).astype(int)

    combined = pd.concat([df_master, new_rows], ignore_index=True)
    fill_cols = [c for c in combined.columns if 'close' in c or 'sma' in c or 'volatility' in c]
    combined[fill_cols] = combined[fill_cols].ffill()
    combined['nvda_volume'] = combined['nvda_volume'].fillna(0).astype(np.int64)
    combined['nvda_return_pct'] = combined['nvda_return_pct'].fillna(0.0)

    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
    combined.to_csv(MASTER_CSV, index=False)
    print(f"Master file updated. Appended {len(new_rows)} rows up to {yesterday}.")

print("--- Pipeline Complete ---")

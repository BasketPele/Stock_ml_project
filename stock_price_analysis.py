import bs4 as bs
import datetime as dt
import numpy as np
import os
import csv
import pandas as pd
import requests
from collections import Counter
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf  

np.random.seed(42)

def save_sp500_tickers_to_csv():
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    
    tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:] if row.find_all('td')]
    
    with open("sp500tickers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['ticker'])
        writer.writerows([[ticker] for ticker in tickers])
    
    print("Tickers saved to 'sp500tickers.csv'")
    return tickers



def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500 or not os.path.exists("sp500tickers.csv"):
        save_sp500_tickers_to_csv()

    tickers = pd.read_csv("sp500tickers.csv")['ticker'].tolist()
    os.makedirs('stock_data', exist_ok=True)

    start, end = dt.datetime(2010, 1, 1), dt.datetime.now()

    for ticker in tickers:
        file_path = f'stock_data/{ticker}.csv'
        if not os.path.exists(file_path):

            df = yf.download(ticker, start=start, end=end)

            if df.empty:
                print(f"Skipping {ticker}: No data available")
                continue


            df.reset_index(inplace=True)


            df.to_csv(file_path, index=False)
            print(f"Downloaded and saved data for {ticker}")
        else:
            print(f"Already have data for {ticker}")



def compile_data():
    tickers = pd.read_csv("sp500tickers.csv")['ticker'].tolist()
    main_df = pd.DataFrame()
    
    for count, ticker in enumerate(tickers):
        file_path = f'stock_data/{ticker}.csv'
        try:
            df = pd.read_csv(file_path, usecols=["Date", "Close"], parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df.rename(columns={"Close": ticker}, inplace=True)
            
            main_df = df if main_df.empty else main_df.join(df, how='outer')
            if count % 10 == 0:
                print(f"Processed {count} tickers")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    main_df.to_csv('sp500_joined_closes.csv')
    print("Compiled data saved to 'sp500_joined_closes.csv'")



def process_data_for_labels(ticker):

    df = pd.read_csv(f'stock_data/{ticker}.csv', header=0)
    

    if df.iloc[1, 0] == 'XEL':
        df = df.drop(index=1)
    

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    if 'Close' not in df.columns:
        print(f"Warning: Column 'Close' not found in the data for {ticker}")
        return None


    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    

    for i in range(1, 8):
        df[f'{ticker}_{i}d'] = (df['Close'].shift(-i) - df['Close']) / df['Close']

    return df



def buy_sell_hold(*args):
    for col in args:
        if col > 0.028: return 1  
        if col < -0.027: return -1  
    return 0  



def extract_featuresets(ticker):
    df = process_data_for_labels(ticker) 
    
    df[f'{ticker}_target'] = df[[f'{ticker}_{i}d' for i in range(1, 8)]].apply(lambda x: buy_sell_hold(*x), axis=1)
    print('Data spread:', Counter(df[f'{ticker}_target'].values.tolist()))
    
    df_vals = df[['Close']].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    X, y = df_vals.values, df[f'{ticker}_target'].values
    return X, y, df



def do_ml(ticker):
    df = process_data_for_labels(ticker)
    if df is None:
        print(f"Skipping {ticker} due to missing data.")
        return
    
    X, y, _ = extract_featuresets(ticker)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = VotingClassifier([
        ('lsvc', svm.LinearSVC(dual=False)),
        ('knn', neighbors.KNeighborsClassifier()),
        ('rfor', RandomForestClassifier())
    ])
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(f'Accuracy for {ticker}:', confidence)
    return confidence



save_sp500_tickers_to_csv()
get_data_from_yahoo()
compile_data()
do_ml('NVDA')

"""
Optimizing Stock Trading Strategy with K-Means Clustering

This script implements a stock market clustering analysis using the K-Means algorithm.
It downloads historical stock data, calculates daily price movements, applies
Principal Component Analysis (PCA) for dimensionality reduction, and groups 
similar companies together using K-Means clustering.

Authors: Amey Thakur & Mega Satish
Institution: Terna Engineering College
Date: April 14, 2021
"""

import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from collections import OrderedDict
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_stock_data(companies_dict, start_date, end_date):
    """
    Downloads historical stock data for a given set of companies.
    """
    logging.info(f"Downloading data from {start_date} to {end_date}...")
    tickers = list(companies_dict.values())
    
    data = []
    # Using tqdm for progress bars as requested
    for ticker in tqdm(tickers, desc="Downloading Stock Data"):
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not stock.empty:
                data.append(stock)
            else:
                logging.warning(f"No data found for {ticker}")
        except Exception as e:
            logging.error(f"Error downloading {ticker}: {e}")
            
    # Concatenate data into a multi-index DataFrame similar to pandas_datareader output
    df = pd.concat(data, axis=1, keys=tickers)
    return df

def analyze_stocks(companies_dict, start_date, end_date):
    """
    Performs clustering analysis on stock movements.
    """
    df = get_stock_data(companies_dict, start_date, end_date)
    
    if df.empty:
        logging.error("No data available for analysis.")
        return

    logging.info("Processing stock movements...")
    
    # Extract Open and Close prices
    # Note: df[ticker] returns the data for that ticker. 
    # We need to reshape the data to (n_companies, n_days)
    
    tickers = list(companies_dict.values())
    valid_tickers = [t for t in tickers if t in df.columns.get_level_values(0)]
    
    # Calculate movements: (Close - Open)
    movements = []
    for ticker in valid_tickers:
        move = df[ticker]['Close'] - df[ticker]['Open']
        movements.append(move.values)
    
    movements = np.array(movements)
    
    # Handle NaN values if any
    if np.isnan(movements).any():
        logging.warning("NaN values found in movements. Filling with zeros.")
        movements = np.nan_to_num(movements)

    logging.info(f"Shape of movement data: {movements.shape}")

    # Normalizer, PCA, and KMeans Pipeline
    normalizer = Normalizer()
    reduced_data = PCA(n_components=2)
    kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=42)
    
    pipeline = make_pipeline(normalizer, reduced_data, kmeans)
    
    logging.info("Fitting K-Means model...")
    pipeline.fit(movements)
    labels = pipeline.predict(movements)
    
    # Results Table
    results = pd.DataFrame({'labels': labels, 'companies': [list(companies_dict.keys())[tickers.index(t)] for t in valid_tickers]})
    results = results.sort_values('labels')
    
    logging.info("\nClustering Results:")
    print(results.to_string(index=False))
    
    # Visualization
    visualize_clusters(movements, pipeline, [list(companies_dict.keys())[tickers.index(t)] for t in valid_tickers])

def visualize_clusters(movements, pipeline, company_names):
    """
    Visualizes the clustering results with decision boundaries.
    """
    logging.info("Generating visualization...")
    
    # Get PCA results separately for plotting
    normalizer = pipeline.named_steps['normalizer']
    pca = pipeline.named_steps['pca']
    kmeans = pipeline.named_steps['kmeans']
    
    norm_movements = normalizer.transform(movements)
    reduced_movements = pca.transform(norm_movements)
    
    # Create meshgrid for decision boundaries
    h = 0.01
    x_min, x_max = reduced_movements[:, 0].min() - 0.1, reduced_movements[:, 0].max() + 0.1
    y_min, y_max = reduced_movements[:, 1].min() - 0.1, reduced_movements[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(reduced_movements[:, 0], reduced_movements[:, 1], 'k.', markersize=5)
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    
    plt.title('K-Means Clustering of Stock Movements (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add labels for companies
    for i, txt in enumerate(company_names):
        plt.annotate(txt, (reduced_movements[i, 0], reduced_movements[i, 1]), size=8)
    
    filename = 'stock_clusters.png'
    plt.savefig(filename)
    logging.info(f"Visualization saved as {filename}")
    plt.show()

if __name__ == "__main__":
    companies_dict = OrderedDict({
        'Amazon':'AMZN',
        'Apple':'AAPL',
        'Walgreen':'WBA',
        'Northrop Grumman':'NOC',
        'Boeing':'BA',
        'Lockheed Martin':'LMT',
        'McDonalds':'MCD',
        'Intel':'INTC',
        # 'Navistar':'NAV', # Removed as it might be delisted or problematic
        'IBM':'IBM',
        'Texas Instruments':'TXN',
        'MasterCard':'MA',
        'Microsoft':'MSFT',
        'General Electrics':'GE',
        # 'Symantec':'SYMC', # Broadcom acquired it, ticker changed
        'American Express':'AXP',
        'Pepsi':'PEP',
        'Coca Cola':'KO',
        'Johnson & Johnson':'JNJ',
        'Toyota':'TM',
        'Honda':'HMC',
        'Mistubishi':'MSBHY',
        # 'Sony':'SNE', # Ticker is now SONY
        'Sony Group':'SONY',
        'Exxon':'XOM',
        'Chevron':'CVX',
        'Valero Energy':'VLO',
        'Ford':'F',
        'Bank of America':'BAC'
    })

    start_date = '2015-01-01'
    end_date = '2017-12-31'
    
    try:
        analyze_stocks(companies_dict, start_date, end_date)
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")

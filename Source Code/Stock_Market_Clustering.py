"""
=============================================================================
Project: OPTIMIZING STOCK TRADING STRATEGY WITH K-MEANS CLUSTERING
Authors: Amey Thakur & Mega Satish
Institution: Terna Engineering College
Date: April 14, 2021
Repository: https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING
Profiles: https://github.com/Amey-Thakur | https://github.com/msatmod
=============================================================================

Stock_Market_Clustering.py
=============================================================================
This production-ready script implements an unsupervised machine learning pipeline 
to identify patterns in stock market movements. Using historical data fetched 
via Yahoo Finance, it classifies companies into clusters based on their daily 
price volatility and movement correlations.

Analytical Pipeline:
1.  Data Acquisition: Retrieval of historical Open/Close prices using yfinance.
2.  Feature Engineering: Calculation of daily price movements (Close - Open).
3.  Normalization: Scaling movement vectors to unit norm for distance-based clustering.
4.  Dimensionality Reduction: Applying PCA to project high-dimensional movement 
    data into a 2D plane for visualization and noise reduction.
5.  Clustering: Execution of the K-Means algorithm to group similar equities.
6.  Visualization: Generation of a cluster plot with decision boundaries.
=============================================================================
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

# Configure terminal logging for real-time execution monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_stock_data(companies_dict, start_date, end_date):
    """
    Downloads historical stock data for a given set of companies using the yfinance API.
    
    Args:
        companies_dict (OrderedDict): Dictionary mapping company names to ticker symbols.
        start_date (str): Beginning of the historical data range (YYYY-MM-DD).
        end_date (str): End of the historical data range (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: A multi-index DataFrame containing historical OHLC data.
    """
    logging.info(f"Initiating stock data acquisition system...")
    logging.info(f"Target Period: {start_date} to {end_date}")
    tickers = list(companies_dict.values())
    
    data = []
    # Implementation of tqdm progress bars for enhanced user experience in CLI
    for ticker in tqdm(tickers, desc="Data Retrieval Progress"):
        try:
            # Fetching data without local caching for fresh analysis
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not stock.empty:
                data.append(stock)
            else:
                logging.warning(f"Warning: No valid records found for ticker symbol: {ticker}")
        except Exception as e:
            logging.error(f"Critical error during retrieval for {ticker}: {str(e)}")
            
    # Concatenate individual datasets into a unified analytical structure
    df = pd.concat(data, axis=1, keys=[t for i, t in enumerate(tickers) if i < len(data)])
    return df

def analyze_stocks(companies_dict, start_date, end_date):
    """
    Orchestrates the end-to-end stock clustering analytical pipeline.
    
    This function handles data processing, model training, and results logging.
    """
    df = get_stock_data(companies_dict, start_date, end_date)
    
    if df.empty:
        logging.error("Pipeline Termination: No data available for analytical processing.")
        return

    logging.info("Starting feature engineering and movement calculation...")
    
    tickers = list(companies_dict.values())
    valid_tickers = [t for t in tickers if t in df.columns.get_level_values(0)]
    
    # Mathematical extraction of Daily Movement: Delta = Close_Price - Open_Price
    # This represents the net intraday performance of each stock.
    movements = []
    for ticker in valid_tickers:
        move = df[ticker]['Close'] - df[ticker]['Open']
        movements.append(move.values)
    
    movements = np.array(movements)
    
    # Data Cleaning: Handling potential NaN values from delisted or restricted tickers
    if np.isnan(movements).any():
        logging.warning("Data Sanitization: Missing values detected. Applying zero-fill imputation.")
        movements = np.nan_to_num(movements)

    logging.info(f"Feature Matrix Construction Complete: {movements.shape[0]} entities across {movements.shape[1]} trading days.")

    # Machine Learning Pipeline Architecture
    # 1. Normalizer: Ensures that stocks with different price levels (e.g., BRK vs AAPL) 
    #    are compared based on percentage-like movements.
    # 2. PCA: Reduces feature noise and maintains variance in a 2D space.
    # 3. KMeans: Groups companies into 10 distinct market-sector-like clusters.
    pipeline = make_pipeline(
        Normalizer(), 
        PCA(n_components=2), 
        KMeans(n_clusters=10, max_iter=1000, random_state=42)
    )
    
    logging.info("Fitting K-Means model for cluster discovery...")
    pipeline.fit(movements)
    labels = pipeline.predict(movements)
    
    # Compilation of clustering results into a human-readable table
    results = pd.DataFrame({
        'Cluster_ID': labels, 
        'Company_Name': [list(companies_dict.keys())[tickers.index(t)] for t in valid_tickers]
    })
    results = results.sort_values('Cluster_ID')
    
    logging.info("\nFinal Clustering Results Summary:")
    print("-" * 40)
    print(results.to_string(index=False))
    print("-" * 40)
    
    # Graphical representation of the resulting clusters
    visualize_clusters(movements, pipeline, [list(companies_dict.keys())[tickers.index(t)] for t in valid_tickers])

def visualize_clusters(movements, pipeline, company_names):
    """
    Generates a 2D scatter plot with decision boundaries to visualize the clusters.
    
    Args:
        movements (np.array): Matrix of stock movements.
        pipeline (sklearn.pipeline.Pipeline): The trained ML pipeline.
        company_names (list): List of company labels corresponding to rows in movements.
    """
    logging.info("Synthesizing graphical visualization...")
    
    # Extract individual components from the pipeline for plotting granularly
    normalizer = pipeline.named_steps['normalizer']
    pca = pipeline.named_steps['pca']
    kmeans = pipeline.named_steps['kmeans']
    
    norm_movements = normalizer.transform(movements)
    reduced_movements = pca.transform(norm_movements)
    
    # Creating a meshgrid to plot the decision regions (color-coded background)
    h = 0.01  # Grid step size
    x_min, x_max = reduced_movements[:, 0].min() - 0.1, reduced_movements[:, 0].max() + 0.1
    y_min, y_max = reduced_movements[:, 1].min() - 0.1, reduced_movements[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predicting cluster for every point in the grid
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(14, 10))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    # Plotting individual companies as black dots
    plt.plot(reduced_movements[:, 0], reduced_movements[:, 1], 'k.', markersize=6)
    
    # Plotting cluster centroids as white cross marks
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=200, linewidths=3,
                color='w', zorder=10)
    
    plt.title('Stock Movement Clustering Analysis\n'
              'Algorithm: K-Means | Dimensionality Reduction: PCA (2-Components)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Annotate company names next to their respective data points
    for i, txt in enumerate(company_names):
        plt.annotate(txt, (reduced_movements[i, 0], reduced_movements[i, 1]), 
                     size=9, alpha=0.9, weight='semibold')
    
    output_filename = 'stock_market_clusters.png'
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Analytical visualization exported successfully: {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Define the equity universe for clustering analysis
    # Structured as OrderedDict to maintain sequential integrity
    market_sectors = OrderedDict({
        'Amazon':'AMZN', 'Apple':'AAPL', 'Walgreen':'WBA', 'Northrop Grumman':'NOC',
        'Boeing':'BA', 'Lockheed Martin':'LMT', 'McDonalds':'MCD', 'Intel':'INTC',
        'IBM':'IBM', 'Texas Instruments':'TXN', 'MasterCard':'MA', 'Microsoft':'MSFT',
        'General Electrics':'GE', 'American Express':'AXP', 'Pepsi':'PEP',
        'Coca Cola':'KO', 'Johnson & Johnson':'JNJ', 'Toyota':'TM', 'Honda':'HMC',
        'Mistubishi':'MSBHY', 'Sony Group':'SONY', 'Exxon':'XOM', 'Chevron':'CVX',
        'Valero Energy':'VLO', 'Ford':'F', 'Bank of America':'BAC'
    })

    # Historical time frame for the analysis
    start_date = '2015-01-01'
    end_date = '2017-12-31'
    
    try:
        analyze_stocks(market_sectors, start_date, end_date)
    except KeyboardInterrupt:
        logging.info("Analysis terminated by user interrupt.")
    except Exception as e:
        logging.critical(f"Pipeline Failure: Unexpected system error encountered: {str(e)}")

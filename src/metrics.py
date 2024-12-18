"""
    this file includes some metrics needed for evaluation
    metrics include:
    · Compound annualized growth
    · Annualized volatility
    · Annualized downside deviation
    · Max drawdown each year
    · Annualized Sharpe Ratio
    · Annualized Sortino Ratio
    · Annualized Omega Ratio
    · Annualized Calmar Ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
from scipy.optimize import minimize
    
def fit_skew_normal(mean, std, skew):
    """
    Fit a skew-normal distribution to match the given mean, std, and skewness.
    
    Parameters:
    - mean (float): Mean of the distribution.
    - std (float): Standard deviation of the distribution.
    - skew (float): Skewness of the distribution.
    
    Returns:
    - params (tuple): Parameters of the skew-normal distribution (alpha, loc, scale).
    """
    # Initial guess for parameters
    initial_alpha = skew
    initial_loc = mean
    initial_scale = std

    # Define the objective function (sum of squared differences of moments)
    def objective(params):
        alpha, loc, scale = params
        fitted_mean = stats.skewnorm.mean(a=alpha, loc=loc, scale=scale)
        fitted_var = stats.skewnorm.var(a=alpha, loc=loc, scale=scale)
        fitted_skew = stats.skewnorm.stats(a=alpha, loc=loc, scale=scale, moments='s')
        
        # Calculate squared errors
        error_mean = (fitted_mean - mean) ** 2
        error_var = (np.sqrt(fitted_var) - std) ** 2
        error_skew = (fitted_skew - skew) ** 2
        
        return error_mean + error_var + error_skew

    # Optimize the parameters
    result = minimize(objective, x0=[initial_alpha, initial_loc, initial_scale],
                      bounds=[(None, None), (None, None), (1e-6, None)],
                      method='L-BFGS-B')
    
    if result.success:
        fitted_alpha, fitted_loc, fitted_scale = result.x
        return fitted_alpha, fitted_loc, fitted_scale
    else:
        raise RuntimeError("Optimization failed. Try different initial parameters or a different distribution.")

def compute_tail_expectations_skew_normal(a, loc, scale, upper_threshold, lower_threshold):
    """
    Calculate both the Expected Tail Return (ETR) and Expected Tail Loss (ETL) 
    for a skew-normal distribution.
    
    Parameters:
    - a (float): Shape parameter (alpha) of the skew-normal distribution.
    - loc (float): Location parameter of the skew-normal distribution.
    - scale (float): Scale parameter of the skew-normal distribution.
    - upper_threshold (float): Threshold for the upper tail (e.g., 95th percentile).
    - lower_threshold (float): Threshold for the lower tail (e.g., 5th percentile).
    
    Returns:
    - etr (float): Expected Tail Return.
    - etl (float): Expected Tail Loss.
    """
    # Define the PDF of the skew-normal distribution
    pdf = lambda x: stats.skewnorm.pdf(x, a, loc=loc, scale=scale)
    
    # Expected Tail Return (ETR)
    # Integral of r * f(r) from upper_threshold to infinity
    integrand_etr = lambda x: x * pdf(x)
    integral_etr, _ = integrate.quad(integrand_etr, upper_threshold, np.inf)
    # Survival function at upper_threshold
    survival = stats.skewnorm.sf(upper_threshold, a, loc=loc, scale=scale)
    etr = integral_etr / survival if survival > 0 else np.nan
    
    # Expected Tail Loss (ETL)
    # Integral of (-r) * f(r) from -infinity to lower_threshold
    integrand_etl = lambda x: x * pdf(x)
    integral_etl, _ = integrate.quad(integrand_etl, -np.inf, lower_threshold)
    # CDF at lower_threshold
    cdf = stats.skewnorm.cdf(lower_threshold, a, loc=loc, scale=scale)
    etl = integral_etl / cdf if cdf > 0 else np.nan  # Negative sign for loss
    
    return etr, etl

### Define GLD Quantile func
def gld_quantile(u, lambda1, lambda2, lambda3, lambda4):
    """
    Quantile function for the Generalized Lambda Distribution (GLD).

    Parameters:
    - u (float or array-like): Probability levels (0 < u < 1).
    - lambda1 (float): Location parameter.
    - lambda2 (float): Scale parameter.
    - lambda3 (float): Shape parameter 1.
    - lambda4 (float): Shape parameter 2.

    Returns:
    - Quantile values corresponding to the probabilities u.
    """
    # Convert u to a NumPy array for element-wise operations
    u = np.asarray(u)
    
    # Ensure that u is within (0, 1)
    if np.any(u <= 0) or np.any(u >= 1):
        raise ValueError("All probability values 'u' must be in the open interval (0, 1).")
    
    return lambda1 + (np.power(u, lambda3) - np.power(1 - u, lambda4)) / lambda2

### Define func for optimisation (GLD):
def gld_mse(params, empirical_quantiles, probs):
    """
    Mean Squared Error between empirical quantiles and GLD quantiles.

    Parameters:
    - params (tuple): GLD parameters (lambda1, lambda2, lambda3, lambda4).
    - empirical_quantiles (array-like): Empirical quantiles.
    - probs (array-like): Corresponding probabilities for the quantiles.

    Returns:
    - mse (float): Mean Squared Error.
    """
    lambda1, lambda2, lambda3, lambda4 = params
    
    # Prevent division by zero or negative scale parameters
    if lambda2 <= 0:
        return np.inf
    
    try:
        gld_q = gld_quantile(probs, lambda1, lambda2, lambda3, lambda4)
    except ValueError as e:
        # If u is out of bounds, return a large MSE
        return np.inf
    
    mse = np.mean((empirical_quantiles - gld_q) ** 2)
    return mse

### Fitting GLD to data
def fit_gld(data, probs=None):
    """
    Fit the Generalized Lambda Distribution (GLD) to the data using quantile matching.

    Parameters:
    - data (array-like): Sample data to fit.
    - probs (array-like, optional): Probabilities for quantiles. Defaults to [0.1, 0.3, 0.7, 0.9].

    Returns:
    - fitted_params (tuple): Estimated GLD parameters (lambda1, lambda2, lambda3, lambda4).
    """
    if probs is None:
        probs = [0.1, 0.3, 0.7, 0.9]  # Default quantiles

    # Calculate empirical quantiles
    empirical_q = np.quantile(data, probs)

    # Initial parameter guesses
    lambda1_init = empirical_q[1]
    lambda2_init = empirical_q[3] - empirical_q[0] if (empirical_q[3] - empirical_q[0]) > 0 else 1.0
    lambda3_init = 1.0
    lambda4_init = 1.0

    initial_params = [lambda1_init, lambda2_init, lambda3_init, lambda4_init]

    # Define bounds to ensure reasonable parameter values
    # lambda2 (scale) > 0, lambda3 and lambda4 > 0
    bounds = [(None, None), (1e-6, None), (1e-6, None), (1e-6, None)]

    # Perform optimization
    result = minimize(
        gld_mse,
        x0=initial_params,
        args=(empirical_q, probs),
        method='L-BFGS-B',
        bounds=bounds
    )

    if result.success:
        fitted_params = result.x
        return tuple(fitted_params)
    else:
        raise RuntimeError("GLD fitting failed. Optimization did not converge.")
    
### Approximate GLD PDF
def approximate_gld_pdf(x, lambda1, lambda2, lambda3, lambda4, num_sim=100000):
    """
    Numerically approximate the PDF of the GLD using simulation and Kernel Density Estimation (KDE).

    Parameters:
    - x (float or array-like): Points at which to evaluate the PDF.
    - lambda1, lambda2, lambda3, lambda4: GLD parameters.
    - num_sim (int): Number of samples to simulate.

    Returns:
    - pdf (float or array-like): Approximated PDF values at x.
    """
    # Simulate samples from the GLD
    u = np.random.uniform(0, 1, num_sim)
    samples = gld_quantile(u, lambda1, lambda2, lambda3, lambda4)

    # Use Gaussian KDE to estimate the PDF
    kde = stats.gaussian_kde(samples)
    return kde.evaluate(x)

### Compute etr etl
def compute_tail_expectations_gld(lambda1, lambda2, lambda3, lambda4, upper_alpha=0.95, lower_beta=0.05, num_sim=100000):
    """
    Calculate both the Expected Tail Return (ETR) and Expected Tail Loss (ETL) 
    for a Generalized Lambda Distribution (GLD) using simulation.

    Parameters:
    - lambda1, lambda2, lambda3, lambda4: GLD parameters.
    - upper_alpha (float): Confidence level for the upper tail (e.g., 0.95).
    - lower_beta (float): Confidence level for the lower tail (e.g., 0.05).
    - num_sim (int): Number of samples to simulate.

    Returns:
    - etr (float): Expected Tail Return.
    - etl (float): Expected Tail Loss.
    """
    # Calculate thresholds
    upper_threshold = gld_quantile(upper_alpha, lambda1, lambda2, lambda3, lambda4)
    lower_threshold = gld_quantile(lower_beta, lambda1, lambda2, lambda3, lambda4)

    # Simulate samples from the GLD
    u = np.random.uniform(0, 1, num_sim)
    samples = gld_quantile(u, lambda1, lambda2, lambda3, lambda4)

    # ETR: Mean of samples >= upper_threshold
    tail_returns = samples[samples >= upper_threshold]
    etr = np.mean(tail_returns) if len(tail_returns) > 0 else np.nan

    # ETL: Mean of -samples <= lower_threshold
    tail_losses = samples[samples <= lower_threshold]
    etl = -np.mean(tail_losses) if len(tail_losses) > 0 else np.nan

    return etr, etl

# empirical plot
def plot_rachev_thresholds(returns, upper_alpha, lower_beta):
    sns.histplot(returns, bins=50, kde=True)
    # sns.histplot(returns, bins=50, stat='density', color='skyblue')
    # sns.kdeplot(returns, color='red')
    upper_thresh = returns.quantile(upper_alpha)
    lower_thresh = returns.quantile(lower_beta)
    plt.axvline(upper_thresh, color='green', linestyle='--', label=f'Upper {upper_alpha*100}% Threshold')
    plt.axvline(lower_thresh, color='red', linestyle='--', label=f'Lower {lower_beta*100}% Threshold')
    plt.legend()
    plt.title('Empirical Rachev')
    plt.show()

# skew-normal
def plot_fitted_distribution_with_tails(returns, a, loc, scale, upper_alpha, lower_beta):
    """
    Plot the empirical returns distribution with the fitted skew-normal distribution and tail thresholds.
    
    Parameters:
    - returns (pd.Series): Series of asset returns.
    - a (float): Shape parameter (alpha) of the skew-normal distribution.
    - loc (float): Location parameter of the skew-normal distribution.
    - scale (float): Scale parameter of the skew-normal distribution.
    - upper_alpha (float): Confidence level for the upper tail.
    - lower_beta (float): Confidence level for the lower tail.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot empirical histogram
    sns.histplot(returns, bins=50, stat='density', label='Empirical', color='skyblue', kde=False)
    
    # Plot fitted skew-normal PDF
    x = np.linspace(returns.min(), returns.max(), 1000)
    pdf_fitted = stats.skewnorm.pdf(x, a, loc=loc, scale=scale)
    plt.plot(x, pdf_fitted, 'r-', label='Fitted Skew-Normal')
    
    # Plot upper and lower thresholds
    upper_threshold = stats.skewnorm.ppf(upper_alpha, a, loc=loc, scale=scale)
    lower_threshold = stats.skewnorm.ppf(lower_beta, a, loc=loc, scale=scale)
    plt.axvline(upper_threshold, color='green', linestyle='--', label=f'Upper {upper_alpha*100}% Threshold')
    plt.axvline(lower_threshold, color='red', linestyle='--', label=f'Lower {lower_beta*100}% Threshold')
    
    plt.legend()
    plt.title('Empirical vs Fitted Skew-Normal Distribution with Tail Thresholds')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.show()

# Plot GLD with Tail Thresholds
def plot_gld_with_tails(returns, lambda1, lambda2, lambda3, lambda4, upper_alpha=0.95, lower_beta=0.05, num_sim=100000):
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, stat='density', label='Empirical', color='skyblue', kde=False)
    x = np.linspace(returns.min(), returns.max(), 1000)
    pdf_fitted = approximate_gld_pdf(x, lambda1, lambda2, lambda3, lambda4, num_sim=num_sim)
    plt.plot(x, pdf_fitted, 'r-', label='Fitted GLD (Approximate PDF)')
    upper_threshold = gld_quantile(upper_alpha, lambda1, lambda2, lambda3, lambda4)
    lower_threshold = gld_quantile(lower_beta, lambda1, lambda2, lambda3, lambda4)
    plt.axvline(upper_threshold, color='green', linestyle='--', label=f'Upper {upper_alpha*100}% Threshold')
    plt.axvline(lower_threshold, color='red', linestyle='--', label=f'Lower {lower_beta*100}% Threshold')
    plt.legend()
    plt.title('Empirical vs Fitted GLD with Tail Thresholds')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.show()

def get_rachev_ratio(returns, upper_q=0.95,  lower_q=0.05, method='gld', quantile_probs = np.arange(0.05,1,0.05), incl_plot=False):
    '''
    This functions returns the Rachev Ratio
    Parameters:
    returns:  pandas Series of %returns
    upper_q (float): The upper quantile of the distribution, used to calculate ETR
    lower_q (float): The lower quantile of the distribution, used to calculate ETL
    method (str): 'gld', 'empirical', skew'; The method used to calculate the Rachev Ratio;
    incl_plot  (bool): Whether to include a plot of the distribution
    
    Returns:
        rachev_ratio (float): The Rachev Ratio
    '''    
    if lower_q > upper_q:
        raise ValueError('Lower quantile cannot be higher than Upper quantile')
    methods = ['gld',  'empirical', 'skew']
    if method not in methods:
        raise ValueError(f'{method} not in allowable methods: {methods}')
    if isinstance(returns, pd.DataFrame):
        if len(returns.columns) > 1:
            raise ValueError('Returns must be a single-column Dataframe or a pandas Series')
        else:
            returns = returns.iloc[:,0]
    elif not isinstance(returns, pd.Series):
        raise ValueError('Returns must be a pandas Series or single-column DataFrame')

    
    ### Empirical Method
    if method == 'empirical':
        upper_threshold = returns.quantile(upper_q)
        lower_threshold = returns.quantile(lower_q)
        etr = returns[returns >= upper_threshold].mean()
        etl = returns[returns <= lower_threshold].mean()
        print(lower_threshold, upper_threshold)
    
    ### Skew-Normal Distribution method
    elif method == 'skew':
        mean_rets = returns.mean()
        std_rets =  returns.std()
        skew_rets = returns.skew()
        # kurt_rets = returns.kurtosis() # Kurtosis not needed
        alpha, loc, scale = fit_skew_normal(mean_rets, std_rets, skew_rets)
        # define thresholds:
        upper_threshold = stats.skewnorm.ppf(upper_q, a=alpha, loc=loc, scale=scale)
        lower_threshold = stats.skewnorm.ppf(lower_q, a=alpha, loc=loc, scale=scale)
        etr, etl = compute_tail_expectations_skew_normal(alpha, loc, scale, upper_threshold, lower_threshold)
        
    ### Generalized Lambda Distribution method
    elif method == 'gld':
        # Step 1: Fit GLD to the data
        fitted_params = fit_gld(returns, probs=quantile_probs)
        lambda1, lambda2, lambda3, lambda4 = fitted_params
        
        # Step 2: Compute ETR and ETL
        etr, etl = compute_tail_expectations_gld(lambda1, lambda2, lambda3, lambda4, upper_q, lower_q)
        etl = -etl
    
    if etl != 0:
        rachev_ratio = etr/-etl
    else:
        rachev_ratio = np.nan
        
    if incl_plot:
        if method == 'empirical':
            plot_rachev_thresholds(returns, upper_q, lower_q)
        elif method == 'skew':
            plot_fitted_distribution_with_tails(returns, alpha, loc, scale, upper_alpha=upper_q, lower_beta=lower_q)
        elif method == 'gld':
             # Plot the Distribution with Tail Thresholds
            plot_gld_with_tails(returns, fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3],
                    upper_alpha=upper_q, lower_beta=lower_q)

    return rachev_ratio
    
def calc_rolling_corr(benchmark, portfolios, window_size, plot=True, highlight_regions=None, benchmark_name='benchmark', plot_path=None, plot_name=None, xlims=None):
    '''
    input:
    benchmark: pd.Series or single col pd.Dataframe; correlation of portfolios will be with respect to this benchmark
    portflios: pd.Dataframe; each column is % ret timeseries
    window_size: int; rolling window size
    plot: bool (default: True); Whether to include a plot
    '''
    
    # Check that benchmark is a series, if single col df, cast to series
    if isinstance(benchmark, pd.Series):
        pass
    elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) == 1:
        benchmark = benchmark.iloc[:, 0]
    else:
        raise TypeError('Expected a pd.Series or a single-column pd.Dataframe for the benchmark')

    # Check that portfolios is a dataframe
    if not isinstance(portfolios, pd.DataFrame):
        raise TypeError('Expected a pd.DataFrame for the portfolios')
    
    # Check that y and x are the same len    
    if len(benchmark.index) != len(portfolios.index):
        raise ValueError(f'Benchmark and Porfolio data do not have the same length; ({len(benchmark.index)} and {len(portfolios.index)})')
    
    rolling_corr = portfolios.rolling(window=window_size).corr(benchmark).dropna()
    
    # For label
    if window_size < 21:
        num = window_size
        units = 'day'
    elif window_size >= 21 and window_size < 252:
        num = round(window_size/21, 2)
        units = 'month'
    else:
        num = round(window_size/252, 2)
        units = 'year'
        
    if plot:
        fig, ax1 = plt.subplots(figsize=(14, 7))  # Explicitly create figure and axis objects
        
        # Plot rolling correlations
        for column in portfolios.columns:
            ax1.plot(rolling_corr[column], label=f'{column}')
        
        ax1.set_title(f'Rolling Correlation with {benchmark.name} ({num} - {units} window)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Correlation')
        ax1.axhline(0, color='black', lw=0.5, ls='--')  # Add a horizontal line at y=0

        # Highlight regions
        if highlight_regions:
            for start_date, end_date in highlight_regions:
                ax1.axvspan(start_date, end_date, color='yellow', alpha=0.5)

        # Calculate cumulative returns for the benchmark
        cumulative_returns = (benchmark + 1).cumprod()
        
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(rolling_corr.index, cumulative_returns.loc[rolling_corr.index], color='red', 
                label=f'Cumulative Returns of {benchmark_name}', linestyle='--')
        ax2.set_ylabel(f'Cumulative Returns', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2
        
        # Place the combined legend at the bottom with no gap
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

        # Adjust bottom margin tightly to the legend
        fig.subplots_adjust(bottom=0.08)  # Reduce bottom spacing
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Tight layout excluding legend space
        
        if xlims:
            ax1.set_xlim(xlims)
            
        if plot_path and plot_name:
            # Save the figure as an image
            fig.savefig(plot_path + f'{plot_name}.png', dpi=300, bbox_inches='tight')
        # else:
        plt.show()
    
    return rolling_corr

if __name__ == '__main__':
    
    FILEPATH = '/Users/gelo/REPOS/Low-Tide-Capital-Management/data/'
    strategy_data =  pd.read_csv(FILEPATH + 'pnl_data/XAUBNG.csv', parse_dates=['Date'], index_col=0)[['equity_change']].rename(columns={'equity_change':'Gold Strategy'})
    base_data =  pd.read_csv(FILEPATH + 'pnl_data/60_40Ports.csv', parse_dates=['Date'], index_col=0)[['ports_pctchange']].rename(columns={'ports_pctchange':'6040 Portfolio'})
    spx_data =  pd.read_csv(FILEPATH + 'stock/SPX.csv', parse_dates=['Date'], index_col=0).pct_change().rename(columns={'Close':'SPX'}).fillna(0)
    gold_data =  pd.read_csv(FILEPATH + 'stock/XAUBNG.csv', parse_dates=['Date'], index_col=0).pct_change().fillna(0).rename(columns={'Close':'XAU'})
    
    # combine data to test
    start = '1975'
    benchmark = base_data.loc[start:]
    portfolios = pd.DataFrame()
    for rets in [strategy_data, gold_data, spx_data]:
        portfolios = pd.concat([portfolios, rets.loc[start:]], axis=1)
    assert len(portfolios) == len(benchmark)
    
    rolling_corr = calc_rolling_corr(benchmark, portfolios, window_size=252*3, benchmark_name='6040 Ports')
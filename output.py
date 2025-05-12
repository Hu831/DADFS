import numpy as np
import pandas as pd

def output(
    DFS_Ya, DFS_d_act, DFS_d_theo, DFS_w, DFS_d_alt,
    DFS_Ya_B, DFS_d_theo_B, DFS_d_act_B, DFS_d_alt_B,
    DFS_HO, count, erra, save, distrib, nens, Nd, r, 
    DA_method, rseed_obs, corr_o, l_o, ylon, ylat
):
    
    """
    Print and optionally save DFS diagnostics and RMSE.

    Parameters
    ----------
    DFS_Ya : ndarray of shape (nobs,)
        Theoretical DFS using ensemble perturbation approach.
    DFS_d_act : ndarray of shape (nobs,)
        Actual DFS (innovation-based).
    DFS_d_theo : ndarray of shape (nobs,)
        Theoretical DFS (innovation-based).
    DFS_w : ndarray of shape (nobs,)
        Actual DFS using weighting vector.
    DFS_d_alt : ndarray of shape (nobs,)
        Theoretical DFS (alternative innovation-based).
    DFS_Ya_B : ndarray of shape (nobs,)
        DFS_Ya estimated using true B.
    DFS_d_theo_B : ndarray of shape (nobs,)
        DFS_d_theo estimated using true B.
    DFS_d_act_B : ndarray of shape (nobs,)
        DFS_d_act estimated using true B.
    DFS_d_alt_B : ndarray of shape (nobs,)
        DFS_d_alt estimated using true B.
    DFS_HO : ndarray of shape (nobs,)
        Theoretical DFS from Hotta & Ota strategy.
    count : ndarray of shape (nobs,)
        Number of grid points each observation is used for.
    erra : ndarray of shape (n,)
        Analysis error at model grid points.
    save : int
        1 to save results to CSV; 0 otherwise.
    distrib : int
        Observation distribution type (1–5).
    nens : int
        Ensemble size.
    Nd : int
        Sample size.
    r : int
        Localization radius.
    DA_method : str
        Data assimilation method: 'LETKF' or 'EnKF'.
    rseed_obs : int
        Random seed used for generating observations.
    corr_o : str
        Observation error correlation type (e.g., 'diag').
    l_o : float
        Correlation length scale (used if corr_o ≠ 'diag').
    ylon, ylat : ndarray of shape (nobs,)
        Longitude and latitude of observations.
    """
    
    # Avoid division by zero by replacing zeros with 1 (for safe averaging)
    safe_count = np.where(count == 0, 1, count)
    
    print('-----')
    print('The DFS for all observations:')
    print(f'DFS_Ya:      {np.sum(DFS_Ya / safe_count):.4f}')
    print(f'DFS_d_act:   {np.sum(DFS_d_act / safe_count):.4f}')
    print(f'DFS_d_theo:  {np.sum(DFS_d_theo / safe_count):.4f}')
    print(f'DFS_w:       {np.sum(DFS_w / safe_count):.4f}')
    print(f'DFS_d_alt:   {np.sum(DFS_d_alt / safe_count):.4f}')
    print(f'DFS_Ya_B:    {np.sum(DFS_Ya_B / safe_count):.4f}')
    print(f'DFS_d_theo_B:{np.sum(DFS_d_theo_B / safe_count):.4f}')
    print(f'DFS_d_act_B: {np.sum(DFS_d_act_B / safe_count):.4f}')
    print(f'DFS_d_alt_B: {np.sum(DFS_d_alt_B / safe_count):.4f}')
    print(f'DFS_HO:      {np.sum(DFS_HO):.4f}')
    print('-----')
    print(f'Analysis RMSE: {np.mean(erra):.4f}')

    # Save data
    if save == 1:

        # Put data into a pandas data frame
        df = pd.DataFrame({
            'lon': ylon,
            'lat': ylat, 
            'Ya': DFS_Ya,
            'w': DFS_w,
            'd_theo': DFS_d_theo, 
            'd_act': DFS_d_act, 
            'd_alt': DFS_d_alt,
            'Ya_B': DFS_Ya_B,
            'd_theo_B': DFS_d_theo_B, 
            'd_act_B': DFS_d_act_B, 
            'd_alt_B': DFS_d_alt_B,
            'HO': DFS_HO,
            'count': count,
            "erra": np.mean(erra)
        })
        
        # Construct filename based on options
        if corr_o == 'diag':
            filename = f'DFS_dist{distrib}_N{nens}_Nd{Nd}_r{r}_{DA_method}_{rseed_obs}.csv'
        else:
            filename = f'DFS_dist{distrib}_N{nens}_Nd{Nd}_r{r}_{DA_method}_{l_o}km_{rseed_obs}.csv'
        
        print('-----')
        print('Saved to:', filename)
        df.to_csv(filename, index=False)
import pickle
import pandas as pd
import os

if os.path.exists('results/original_result.p'):
    print('Displaying paper results:')
    df_paper_result = pickle.load(open('results/original_result.p', 'rb'))
    print(df_paper_result.head())
if os.path.exists('results/final_results.p'):
    print('Displaying calculated results:')
    df_own_result = pickle.load(open('results/final_results.p', 'rb'))
    print(df_own_result.head())
elif os.path.exists('results/results_dict.p'):
    print('Displaying intermediate results:')
    df_data = pickle.load(open('results/results_dict.p', 'rb'))
    res = pd.DataFrame(data=df_data)
    print(res.head())
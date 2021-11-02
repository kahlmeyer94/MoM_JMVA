import numpy as np
from tqdm import tqdm
import pandas as pd
import utils
import torch
import os
import mom
from timeit import default_timer as timer
import dotted_dict
from dotted_dict import DottedDict


'''
Synthetic Experiment
'''
def synth_exp(ns, nmb_models, dest_dir, dims, ts, alpha0s, sample_alphas, sample_mus, sample_sigmas, sigma_eps= 1e-3, seed = None, save_intermediate=False, *args, **kwargs):
    '''
    Synthetic Experiment for ICML

    @Params:
        ns...                   number of data for learning (list)
        nmb_models...           number of models in population
        dest_dir...             directory where results are saved
        dim...                  list of feature dimensions
        t...                    list of number of topics
        alpha0...               list of alpha0s
        sample_alphas           function on alpha_0,t of how to sample alphas
        sample_mus              function on c,t of how to sample mus
        sample_sigmas           function on sigma_eps,t of how to sample sigmas
        sigma_eps...            [optional] Minimum value for sigmas 
        seed...                 [optional] Seed for reproducibility
        save_intermediate...    [optional] if true, saves dataframe after each model

    @Returns:
        pandas dataframe with results
    '''
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # draw population of models
    print('Sampling Models...', end="")
    models = []
    for c in dims:
        for t in ts:
            for alpha_0 in alpha0s:
                for _ in range(nmb_models):
                    alphas = torch.as_tensor(sample_alphas(alpha_0,t))
                    mus = torch.as_tensor(sample_mus(c, t))
                    sigmas = torch.as_tensor(sample_sigmas(sigma_eps, t))

                    model = DottedDict()
                    model.c = c
                    model.alphas = alphas
                    model.mus = mus
                    model.sigmas = sigmas
                    models.append(model)
    print('done')
    
    print('Running experiment')
    df_data = {'#samples': [], 't': [], 'c': [], 'alpha0': [], 'error mu': [], 'error alpha': [], 'error var': [], 'time': []}
    for model in tqdm(models):
        # draw dataset of maximum size
        trainset, train_labels = utils.generate_data(model.mus, model.sigmas, model.alphas, max(ns), labels=True)

        # apply mom on partial datasets
        for n in ns:
            part_data, part_labels = trainset[:n], train_labels[:n]
            c, t = model.mus.shape
            alpha0 = torch.sum(model.alphas).item()
            orig_mus = model.mus
            orig_sigmas = model.sigmas
            orig_alphas = model.alphas

            # measure time
            m = mom.MoM(alpha0, t, c, timing=True)
            s_time = timer()
            m.fit(part_data)
            e_time = timer()
            times = m.times
            times.append(e_time-s_time)
            df_data['time'].append(times)

            # match results
            order = utils.get_matching([model.mus.T],[m.est_thetas.T], utils.cost_func)
            est_alphas = m.est_alphas[order]
            est_vars = m.est_vars[order]
            est_mus = m.est_thetas[:,order]
            est_vars[est_vars<=0] = sigma_eps**2
            m.est_alphas = est_alphas
            m.est_vars = est_vars
            m.est_thetas = est_mus

            # measure error
            mu_error = torch.mean((est_mus-orig_mus)**2).item()
            var_error = torch.mean((est_vars-orig_sigmas)**2).item()
            alpha_error = torch.mean((est_alphas-orig_alphas)**2).item()

            # put into dataframe + save
            df_data['#samples'].append(n)
            df_data['t'].append(t)
            df_data['c'].append(c)
            df_data['alpha0'].append(alpha0)
            df_data['error mu'].append(mu_error)
            df_data['error var'].append(var_error)
            df_data['error alpha'].append(alpha_error)

            if save_intermediate:
                p = os.path.join(dest_dir, 'results_dict.p')
                utils.save_data(df_data, p)


    res = pd.DataFrame(data=df_data)
    p = os.path.join(dest_dir, 'final_result.p')
    utils.save_data(res, p)
    return res


if __name__ == "__main__":
    
    params = {
        # number of data
        'ns' : [2000,10000,20000,50000],
        # directory where results are saved
        'dest_dir' : 'results',
        # minimum value for sigmas 
        'sigma_eps' : 1e-3,
        # seed for reproducibility
        'seed' : 0,
        # saving results
        'save_intermediate' : True,
        # number of models per parameters setting
        'nmb_models' : 5,
        # sampling of parameters
        'dims' : [128,256,512,1024,2048],
        'ts' : [30,50,70,90],
        'alpha0s' : [0.1,0.5,1,5,10,50,100],
        'sample_alphas' : lambda alpha_0,t: utils.sample_uniform_alphas(t, alpha_0),
        'sample_mus' : lambda c,t: utils.create_pairwise_independent_mus(t, c, vmin=-10, vmax=10).T,
        'sample_sigmas' : lambda sigma_eps,t: np.random.uniform(low=sigma_eps, high=2, size=(t,))
        
    }

    df_result = synth_exp(**params)
    df_result.head()
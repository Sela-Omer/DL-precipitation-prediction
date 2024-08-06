#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

# ## Imports

# In[2]:


import torch
from src.config.config import config
from src.service.service_data_analysis import ServiceDataAnalysis

# ## Define parameters to explore

# In[3]:


years = [str(y) for y in range(1945, 2025)]
# PARAMETERS = "intensity,date"
PARAMETERS = "date#sin,date#cos,lat,lon#sin,lon#cos,sp,t2m,intensity,u10,v10,tcw,z,ta_250,ta_300,ta_500,ta_850,ua_250,ua_300,ua_500,ua_850,va_250,va_300,va_500,va_850,za_250,za_300,za_500,za_850,msl,skt,slhf,tp"
config['DATA']['APPLY_TFMS'] = 'correct_intensity,circular_unfold_date,circular_unfold_lon'
config['APP']['MODE'] = 'DATA_ANALYSIS'
config['APP']['BATCH_SIZE'] = '100'
config['DATA']['LOOKBACK_RANGE'] = '3'
config['DATA']['FORECAST_RANGE'] = '0'
config['DATA']['TARGET_PARAMETERS'] = 'intensity'
config['DATA']['INPUT_PARAMETERS'] = PARAMETERS
config['DATA']['PARAMETERS'] = PARAMETERS
config['DATA']['VAL_RATIO'] = '0.01'
config['DATA']['YEARS'] = ",".join([str(y) for y in range(1945, 2025)])
config['DATA']['PATH'] = '/home/mansour/ML3300-24a/omersela3/fixed_tensors-v2/fixed_tensors-v2'
config['APP']['ARCH'] = 'DATA_ANALYSIS_NOTEBOOK'
config['APP']['NETWORK_DEPTH'] = '1'
config['APP']['OVERWRITE_CONFIG_PATH'] = 'DATA_ANALYSIS_NOTEBOOK'


# ## Generate Stats for each year and plot

# ### Generate stats for each year

def generate_stats(year):
    try:
        config['DATA']['YEARS'] = f'{year},'
        service = ServiceDataAnalysis(config)
        simple_script = service.scripts['SIMPLE_NN']
        datamodule = simple_script.create_datamodule()
        datamodule.prepare_data()
        datamodule.setup(stage='fit')

        train_dl = datamodule.train_dataloader()

        # Generate statistics for the audio data in the dataloader
        mean, std, grad_mean, grad_std = simple_script.generate_stats_dataloader(train_dl, exec_grad_stats=True)

        torch.save(mean, f'stats/DATA_ANALYSIS-mean_{year}.pt')
        torch.save(std, f'stats/DATA_ANALYSIS-std_{year}.pt')
        torch.save(grad_mean, f'stats/DATA_ANALYSIS-grad_mean_{year}.pt')
        torch.save(grad_std, f'stats/DATA_ANALYSIS-grad_std_{year}.pt')
    except:
        print(f'failed for year: {year}')
        pass

# execute generate_stats for each year in years using multiprocessing
import multiprocessing
if __name__ == '__main__':
    pool = multiprocessing.Pool(10)
    pool.map(generate_stats, years)
    pool.close()
    pool.join()

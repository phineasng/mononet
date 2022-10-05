import numpy as np
import pandas as pd

import csv
from experimental_code.tabular_data.legacy.MonoNet_class import *
import os


def generate_interv(data_, model_, mu=1, sigma=0.5, nb_sample_interv=50):
    # biomarkers_data = data_.X_data[0:1400, :]
    biomarkers_data = data_.X_data[0:700, :]
    nb_without_interv = biomarkers_data.shape[0]-nb_sample_interv*biomarkers_data.shape[1]
    list_regimes = []
    list_interv = []

    for i in range(biomarkers_data.shape[1]):
        s = np.random.normal(mu, sigma, nb_sample_interv)
        biomarkers_data[nb_sample_interv*i:nb_sample_interv*(i+1), i] = torch.from_numpy(s)
        list_regimes += [i+1]*nb_sample_interv
        list_interv += [i+1]*nb_sample_interv
    list_regimes += [0]*nb_without_interv
    list_interv += [None] * nb_without_interv

    h_val = model_.unconstrainted_block(biomarkers_data)
    biomarkers_h_values = torch.hstack((biomarkers_data, h_val))

    if not os.path.exists('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/'):
        os.makedirs('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/')

    with open('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/intervention1.csv', 'w', newline='', ) as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([[el] for el in list_interv])

    df_regimes = pd.DataFrame(list_regimes)
    df_regimes.to_csv('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/regime1.csv', index=False, header=False)

    np.save('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/data1.npy', biomarkers_h_values.detach().numpy())
    mat_random = np.random.randint(2, size=(21, 21))
    np.fill_diagonal(mat_random, 0)
    np.save('/Users/dam/Documents/GitHub/dcdi/data/perfect/MonoNet_intervention/DAG1.npy',
            mat_random)

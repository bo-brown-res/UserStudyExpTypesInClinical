import os

import torch
import torch.nn
from tensorflow.keras.models import load_model, Model

from CUI020_seq_loader import EmbeddingSequence
from CUI040_prediction import read_data, create_datasets
import CUI041_prediction_config
import numpy as np
import pandas as pd
import time
import tensorflow as tf

POSTIVE_THRESHOLD = 0.065 #based on what is top 5% of risk

class TheWrapper(torch.nn.Module):
    def __init__(self, dat_dict, background_sizes, col_names, to_explain_shape, n_workers=1, multiproc=False, returntensor=False, return_raw=False):
        super(TheWrapper, self).__init__()
        print("initing model...")
        
        self.model_config = read_data(CUI041_prediction_config.config['config_pickle'])
        self.outcome = self.model_config['outcome']
        self.time_var = self.model_config['time_var']
        self.study_id = self.model_config['study_id']
        
        self.dat_dict = dat_dict
        self.pred_vars = dat_dict['pred_vars']
        self.test_encs = dat_dict['test_encs'] 

        custom_obs = read_data(CUI041_prediction_config.config['model_objects'])
        self.model = load_model(CUI041_prediction_config.config['model_file'], custom_objects=custom_obs)
        self.transformer = read_data(CUI041_prediction_config.config['transformer'])
        
        self.background_sizes = background_sizes
        self.col_names = col_names
        self.to_explain_shape = to_explain_shape
        
        self.sort_dict = pd.DataFrame(self.test_encs).reset_index()
        self.sort_dict = self.sort_dict.set_index(0)
        self.sort_dict = self.sort_dict.to_dict()['index']
        
        print(f"IS GPU AVAILABLE: {tf.test.is_gpu_available()}")
        
        #CAREFUL
        self.GPU_TO_USE = 2
        self.n_workers = n_workers
        self.multiproc = multiproc
        self.return_tensor = returntensor
        self.return_raw = return_raw
        

    def forward(self, dat):
        with tf.GradientTape() as t:
            with t.stop_recording():
                if self.to_explain_shape is not None and self.to_explain_shape[-1] == 57:
                    print(f"DOING ADD COLS")
                    dat = dat.reshape([-1] + list(self.to_explain_shape))
                    num_items = dat.shape[0]
                    tensor_encid = torch.ones(list(dat.shape[:-1]))
                    id_arr = torch.arange(tensor_encid.shape[0]).unsqueeze(-1)
                    tensor_encid = (tensor_encid * id_arr).unsqueeze(-1)
                    tensor_cuis = torch.ones_like(tensor_encid)
                    
                    if isinstance(dat, np.ndarray):
                        dat = np.concatenate([tensor_encid.numpy(), dat, tensor_cuis.numpy()], axis=-1)
                    elif isinstance(dat, torch.Tensor):
                        dat = torch.cat([tensor_encid, dat, tensor_cuis], dim=-1)

                    dat = dat.reshape(num_items, -1)

                if dat.shape[0] >= 15000:
                    print(f"HIJACKING PERMUTE {dat.shape} with uniques {np.unique(dat[:,0]).shape}")
                    for ij, perm_sample in enumerate(dat):
                        perm_sample[:, 0] = ij #+ np.random.randint(low=-99999999, high=99999999)
                        dat[ij] = perm_sample
                    print(f"HIJACKING POST uniques {np.unique(dat[:,0]).shape}")

                ################################################################################################
                #### REFORMATTING DATA TO PD ARRAY FOR PROCESSING

                if type(dat) == torch.Tensor:
                    dat = dat.detach().numpy()

                last_dim_size = dat.shape[-1]
                pd_dat = []
                for block in dat:
                    try:                
                        if dat.shape[0] == 1:
                            partial_block = block[block!=-999].reshape(self.to_explain_shape)
                        else:
                            partial_block = block[block!=-999].reshape(-1, 59)
                    except Exception as ex:
                        print(f"IMPORTANT self.to_explain_shape {self.to_explain_shape} with exception: {ex}")
                        partial_block = block[:self.to_explain_shape[0], :self.to_explain_shape[1]]
                        partial_block[partial_block == -999] = -1

                    newblock = pd.DataFrame(partial_block, columns=self.col_names)
                    pd_dat.append(newblock)

                dat = pd.concat(pd_dat, axis=0).reset_index()
                dat['cui'] = [[float(x)] for x in dat.index]

                ################################################################################################
                #### READ IN AND PREP STRUCTURED DATA

                dat.drop(self.model_config['drop_cols'], axis=1, inplace=True, errors='ignore')

                if self.model_config['model_type'] == 'cui':
                    X = dat[[self.study_id, self.time_var, 'cui']]
                    y = dat[[self.study_id, self.time_var, self.outcome]]
                    cuis = self.dat_dict['cuis']
                elif self.model_config['model_type'] == 'ts':
                    nonTrans_dat = dat[[self.study_id, self.time_var]]
                else:
                    nonTrans_dat = dat[[self.study_id, self.time_var, 'cui']]
                    cuis = self.dat_dict['cuis']

                pred_vars = [v for v in self.pred_vars if v not in self.model_config['drop_cols']]

                ################################################################################################
                ################### Transforming data
                X_predictors = np.array(dat[[*pred_vars]]).astype(np.float32)

                if self.model_config['transformation'] is not None:

                    Xtrans = self.transformer.transform(X_predictors)  # transform all data
                else:
                    Xtrans = X_predictors
                X = pd.concat([nonTrans_dat, pd.DataFrame(Xtrans)], axis=1)
                y = dat[[self.study_id, self.time_var, self.outcome]]

                listids = dat['encounter_id'].unique()

                ################################################################################################
                ################### Preparing data
                X_test = X
                y_test = y

                ################################################################################################
                ################### Reading in model and making predictions

                info_df = y_test[[self.study_id, self.time_var]]
                
                fake_pred = np.zeros((info_df.shape[0], 1))
                fake_pred[:,0] = 1
                
                test_generator = EmbeddingSequence(X_test, y_test, listids, max_cui_len=None,
                                                   max_ts_len=2000, batch_size=1, shuffle=False, config=self.model_config)
                try:
                    y_pred = self.model.predict(test_generator, workers=self.n_workers, use_multiprocessing=self.multiproc)
                except Exception as Ex:
                    print(f"FAILURE ON GRAD ACCUM!\n{Ex}\n")
                    print(f"X_test.shape {X_test.shape} and info: {info_df.shape} and fake: {fake_pred.shape}")
                    y_pred = fake_pred
                    
                y_preds = np.vstack(y_pred)
                y_preds = pd.DataFrame(y_preds, columns=['y_pred_prob'])
                
                info_df['test_index'] = info_df.groupby(self.study_id).cumcount()
                info_df = info_df.loc[info_df.test_index < 2000][[self.study_id, self.time_var]]
                info_df.reset_index(drop=True, inplace=True)

                y_preds_df = info_df.merge(y_preds, left_index=True, right_index=True)
    
                if self.return_raw:
                    return y_preds_df

                final_ys = []
                for encid in listids:
                    target_loc = -1 #default is last value
                    matching_preds = y_preds_df[y_preds_df['encounter_id'] == encid]
                    final_ys.append(matching_preds.iloc[target_loc]['y_pred_prob'])

                threshresults = np.zeros((len(final_ys),2))
                for inx, fy in enumerate(final_ys):
                    threshresults[inx, 0] = (POSTIVE_THRESHOLD*2)-fy
                    threshresults[inx, 1] = fy

                if self.return_tensor:
                    threshresults = torch.from_numpy(threshresults)
                    
                return threshresults


    def predict(self, dat):
        return self.forward(dat)
    
    
    def predict_label(self, dat):
        logits = np.array(self.forward(dat))
        pred = np.argmax(logits, axis=1)
        return pred
    
    
    
    
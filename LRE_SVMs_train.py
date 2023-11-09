import numpy as np
import os
import joblib

def binary_model_path(param):
    file = f"{param['model_path']}C{param['svm_C']}W{param['exemplar_weight']}L{param['lambda1']}-{param['lambda2']}c{param['cate_i']}-{param['cate_j']}.model.mat"
    return file

def LRE_SVMs_train(data, param):
    train_ftr = data['ftr']
    train_lbl = data['lbl']
    trn_smpl_num, ftr_dim = train_ftr.shape
    cate_num = np.max(train_lbl)

    weights = np.zeros_like(train_ftr)
    bias = np.zeros(trn_smpl_num)
    for cate_i in range(1, cate_num + 1):
        print(f'LRE_SVMs training... : category : {cate_i}')
        cate_idx = (train_lbl == cate_i)
        tmp_trn_lbl = 2 * cate_idx.astype(int) - 1
        
        param['cate_i'] = cate_i
        param['cate_j'] = 0
        tmp_file = binary_model_path(param)
        if param['save_model_flag'] == 2:
            if os.path.isfile(tmp_file):
                tmp_model = joblib.load(tmp_file)
                print(f'load model from: {tmp_file}')
            else:
                print('no model: creating...')
                tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param)
                joblib.dump(tmp_model, tmp_file)
                print(f'save model to: {tmp_file}')
        elif param['save_model_flag'] == 1:
            tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param)
            joblib.dump(tmp_model, tmp_file)
            print(f'save model to: {tmp_file}')
        elif param['save_model_flag'] == 0:
            tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param)
        else:
            print(f'unknown save_model_flag: {param["save_model_flag"]}')
        
        weights[cate_idx, :] = tmp_model['weights']
        bias[cate_idx] = tmp_model['bias']
    
    model = {
        'esvm': {
            'esvm_weights': weights,
            'esvm_bias': bias,
            'train_lbl': train_lbl
        },
        'cate_num': cate_num,
        'trn_smpl_num': trn_smpl_num,
        'dam_flag': param['dam_flag'],
        'prdct_top_num': param['prdct_top_num'],
        'dam_C': param['dam_g1'],
        'dam_lambda': param['dam_g2'],
        'dam_eps': param['dam_eps']
    }

    return model

# Usage of the function:
# Assuming data and param are defined:
# model = LRE_SVMs_train(data, param)

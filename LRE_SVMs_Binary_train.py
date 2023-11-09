import numpy as np
import os
from scipy.io import loadmat
from scipy.sparse import eps
import math
from scipy.optimize import minimize
from scipy.linalg import svd
import time


def LRE_SVMs_binary_train(src_ftr, src_lbl, param):
    min_val = float(np.finfo(np.float32).eps)
    
    trn_smpl_num, ftr_dim = src_ftr.shape
    pos_idx = (src_lbl == 1)
    pos_num = np.sum(pos_idx)
    neg_idx = (src_lbl == -1)
    neg_num = np.sum(neg_idx)
    
    pos_aug_src_ftr = np.hstack((src_ftr[pos_idx, :], np.ones((pos_num, 1))))
    neg_aug_src_ftr = np.hstack((src_ftr[neg_idx, :], np.ones((neg_num, 1))))
    del src_ftr
    
    init_param = param.copy()
    init_param['lambda1'] = 0
    init_param['lambda2'] = 0
    
    tmp_file = binary_model_path(init_param)
    if os.path.isfile(tmp_file):
        tmp_model = loadmat(tmp_file)['tmp_model']
        print('init from: ' + tmp_file)
        aug_weight = np.hstack((tmp_model['weights'], tmp_model['bias']))
    else:
        max_val = np.maximum(np.sum(pos_aug_src_ftr * pos_aug_src_ftr, axis=1), min_val)
        aug_weight = pos_aug_src_ftr / max_val[:, None]
        tid = time.time()
        aug_weight, out_init_param = find_min_svt(logistic_loss, aug_weight, None, pos_aug_src_ftr, neg_aug_src_ftr, pos_aug_src_ftr, init_param)
        time_taken = time.time() - tid
        print(f'init: lambda = 0, positive samples: {pos_num}, W/F ite : {out_init_param["w_f_ite"]}, objective value: {out_init_param["obj_val"]}, time(s): {time_taken}')
    
    tid = time.time()
    aug_weight, out_param = find_min_svt(logistic_loss, aug_weight, None, pos_aug_src_ftr, neg_aug_src_ftr, pos_aug_src_ftr, param)
    time_taken = time.time() - tid
    print(f'positive samples: {pos_num}, W/F ite : {out_param["w_f_ite"]}, objective value: {out_param["obj_val"]}, time(s): {time_taken}')
    
    model = {
        'weights': aug_weight[:, :ftr_dim],
        'bias': aug_weight[:, ftr_dim]
    }
    
    return model


def find_min_svt(loss_fun, weight, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, param):
    pos_num, ftr_dim = pos_src_ftr.shape
    neg_num, _ = neg_src_ftr.shape
    trsf_num, _ = trsf_ftr.shape

    lambda1 = param['lambda1']
    lambda2 = param['lambda2']
    lambda_flag = lambda1 != 0 or lambda2 != 0
    C2 = param['svm_C']
    C1 = C2 * neg_num * (-param['exemplar_weight']) if param['exemplar_weight'] < 0 else C2 * param['exemplar_weight']
    mu = 1

    if lambda_flag:
        fmat = exemplar_logistic_prob(np.dot(weight, trsf_ftr.T))
        tmp_u, tmp_S, tmp_v = svd(fmat, full_matrices=False)
        sngl_val = np.diag(tmp_S)
        tr_val = np.sum(sngl_val)
    else:
        fmat = np.zeros((pos_num, trsf_num))
        tr_val = 0
        print('lambda =0 : do not need fmat')

    obj_val = float('inf')
    obj_val_new = float('inf')
    cnvg_flag = 0
    w_min_flag = 0
    update_f_flag = 1

    weight_vec = weight.flatten()
    svt_thred = lambda1 / lambda2 / 2 if lambda2 > 0 else 0
    if lambda2 > 0:
        print(f'svt thred: {svt_thred}')

    for ite_i in range(1, param['max_ite'] + 1):

        # given F, update W
        update_w_flag = 0
        if update_f_flag or w_min_flag <= 0:
            start_time = time.time()
            res = minimize(lambda w: weight_fun(loss_fun, w, fmat, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, pos_num, ftr_dim, C1, C2, lambda2, mu),
                           weight_vec, method='L-BFGS-B', options={'disp': None, 'maxiter': param['max_inner_ite']})
            obj_val_new = res.fun + lambda1 * tr_val
            if obj_val_new < obj_val - param['min_obj_val']:
                weight_vec = res.x
                obj_val = obj_val_new
                weight = np.reshape(weight_vec, (pos_num, ftr_dim))
                gmat = exemplar_logistic_prob(np.dot(weight, trsf_ftr.T))
                update_w_flag = 1
                elapsed_time = time.time() - start_time

        if not lambda_flag:
            cnvg_flag = w_min_flag > 0
            if update_w_flag and not cnvg_flag:
                continue
            else:
                break

        # given W, update F
        update_f_flag = 0
        if update_w_flag:
            start_time = time.time()
            try:
                tmp_u, tmp_S, tmp_v = svd(gmat, full_matrices=False)
                tmp_S = np.diag(np.maximum(np.diag(tmp_S) - svt_thred, 0))
                fmat = np.dot(np.dot(tmp_u, tmp_S), tmp_v)
                tr_val = np.sum(np.diag(tmp_S))
                if np.mean(np.abs(fmat - gmat)) < param['min_f_thred']:
                    break
                else:
                    update_f_flag = 1
            except np.linalg.LinAlgError:
                print('svd error')
                update_f_flag = 0
            elapsed_time = time.time() - start_time

        if not update_w_flag and not update_f_flag:
            cnvg_flag = w_min_flag > 0
            break

    weight = np.reshape(weight_vec, (pos_num, ftr_dim))
    out_param = {
        'obj_val': obj_val,
        'w_f_ite': ite_i,
        'cnvg_flag': cnvg_flag
    }
    
    return weight, out_param

def logistic_loss(aug_weight, pos_aug_src_ftr, neg_aug_src_ftr):
    max_val = np.finfo(float).max / 2
    pos_loss = np.minimum(np.exp(-np.sum(aug_weight * pos_aug_src_ftr, axis=1)), max_val)
    neg_loss = np.minimum(np.exp(np.dot(aug_weight, neg_aug_src_ftr.T)), max_val)

    obj_loss = {'pos_loss': np.log(1 + pos_loss), 'neg_loss': np.log(1 + neg_loss)}
    det_loss = {'pos_loss': pos_loss / (1 + pos_loss), 'neg_loss': neg_loss / (1 + neg_loss)}

    return obj_loss, det_loss

def binary_model_path(param):
    """
    Generate the path for the binary model file.
    """
    file = (
        f"{param['model_path']}"
        f"C{param['svm_C']}"
        f"W{param['exemplar_weight']}"
        f"L{param['lambda1']}-{param['lambda2']}"
        f"c{param['cate_i']}-{param['cate_j']}"
        ".model.mat"
    )
    return file

def exemplar_logistic_prob(esvm_predict_val):
    """
    Calculate the logistic probability based on the esvm predict values.
    """
    prob = 1.0 / (1.0 + np.exp(-esvm_predict_val))
    return prob

def weight_fun(loss_fun, weight_vec, fmat, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, pos_num, ftr_dim, C1, C2, lambda2, mu):
    weight = weight_vec.reshape(pos_num, ftr_dim)
    obj_loss, det_loss = loss_fun(weight, pos_src_ftr, neg_src_ftr)

    gmat = exemplar_logistic_prob(np.dot(weight, trsf_ftr.T))

    # Assuming weight_obj and weight_fmat_fun are other functions you have implemented
    obj = (weight_obj(weight, C1, C2, tgt_cor, obj_loss['pos_loss'], obj_loss['neg_loss'], mu) +
           weight_fmat_fun(gmat, fmat) * lambda2)

    # Assuming weight_det and weight_fmat_det are other functions you have implemented
    det = (weight_det(weight, ftr_dim, C1, C2, pos_src_ftr, neg_src_ftr, tgt_cor, det_loss['pos_loss'], det_loss['neg_loss'], mu) +
           weight_fmat_det(trsf_ftr, gmat, fmat) * 2 * lambda2)
    det_vec = det.flatten()
    
    return obj, det_vec

def fmat_det(fmat, gmat):
    return fmat - gmat

def weight_fmat_fun(gmat, fmat):
    diff = gmat - fmat
    return np.dot(diff.ravel(), diff.ravel())

def weight_fmat_det(trsf_ftr, gmat, fmat):
    # The commented line corresponds to a different formula which might be an alternative or previous version.
    # If you need to use that version, just uncomment it and comment the current return statement.
    # det = np.dot((gmat ** 2 * emat * (gmat - fmat)), trsf_ftr)
    det = np.dot((gmat * (1 - gmat) * (gmat - fmat)), trsf_ftr)
    return det

def weight_obj(weight, C1, C2, tgt_cor, pos_loss, neg_loss, mu):
    obj = mu * np.sum(weight ** 2) + C1 * np.sum(pos_loss) + C2 * np.sum(neg_loss)
    return obj

def weight_det(weight, ftr_dim, C1, C2, pos_src_ftr, neg_src_ftr, tgt_cor, pos_loss, neg_loss, mu):
    det = 2 * mu * weight - C1 * pos_src_ftr * np.tile(pos_loss[:, np.newaxis], (1, ftr_dim)) + C2 * np.dot(neg_loss, neg_src_ftr)
    return det

def logistic_loss(aug_weight, pos_aug_src_ftr, neg_aug_src_ftr):
    max_val = np.finfo(float).max / 2
    pos_loss = np.minimum(np.exp(-np.sum(aug_weight * pos_aug_src_ftr, axis=1)), max_val)
    neg_loss = np.minimum(np.exp(np.dot(aug_weight, neg_aug_src_ftr.T)), max_val)

    obj_loss = {'pos_loss': np.log(1 + pos_loss), 'neg_loss': np.log(1 + neg_loss)}
    det_loss = {'pos_loss': pos_loss / (1 + pos_loss), 'neg_loss': neg_loss / (1 + neg_loss)}

    return obj_loss, det_loss

def f_list2mat(theta, f_list, pos_num, trsf_num):
    fmat = np.zeros((pos_num, trsf_num))
    assert len(f_list) == len(theta)
    for w_i in range(len(theta)):
        fmat += theta[w_i] * f_list[w_i]['u'] @ f_list[w_i]['v']
    return fmat
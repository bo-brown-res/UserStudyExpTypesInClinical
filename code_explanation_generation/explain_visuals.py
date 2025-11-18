import os
import pickle

import matplotlib
import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist, bar


def process_counterfactual(batch, **kwargs):
    raw_data = []
    raw_idxs = []
    raw_lbls = []
    for x in batch:
        cf = x[0][0]
        cf_label = x[0][1].item()
        loc = x[1]

        raw_data.append(cf)
        raw_idxs.append(loc)
        raw_lbls.append(cf_label)

    preprocess = kwargs['preprocess']
    counterfactuals = np.stack(raw_data).squeeze()
    if preprocess:
        counterfactuals = preprocess(counterfactuals)

    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()


    saved_metrics = {}
    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    counterf_confidence = model(torch.from_numpy(counterfactuals).to(torch.float32))

    diff_model_confidence = orig_confidence - counterf_confidence
    batch_avg_diff_confidence = np.mean(diff_model_confidence.detach().numpy(), axis=0)
    batch_std_diff_confidence = np.std(diff_model_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)

    batch_cf_std_for_timepoint_for_feature = np.std(counterfactuals, axis=0)

    diff_between_cf_and_orig = orig_inputs - counterfactuals
    batch_mean_diff_along_timepoints = np.mean(diff_between_cf_and_orig, axis=1)
    batch_mean_diff_along_features = np.mean(diff_between_cf_and_orig, axis=2)

    batch_median_diff_along_timepoints = np.median(diff_between_cf_and_orig, axis=1)
    batch_median_diff_along_features = np.median(diff_between_cf_and_orig, axis=2)

    saved_metrics['batch_avg_diff_confidence'] = batch_avg_diff_confidence
    saved_metrics['batch_std_diff_confidence'] = batch_std_diff_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_cf_std_for_timepoint_for_feature'] = batch_cf_std_for_timepoint_for_feature
    saved_metrics['batch_mean_diff_along_timepoints'] = batch_mean_diff_along_timepoints
    saved_metrics['batch_mean_diff_along_features'] = batch_mean_diff_along_features
    saved_metrics['batch_median_diff_along_timepoints'] = batch_median_diff_along_timepoints
    saved_metrics['batch_median_diff_along_features'] = batch_median_diff_along_features
    saved_metrics['og_mean'] = np.mean(orig_inputs, axis=0)
    saved_metrics['cf_mean'] = np.mean(counterfactuals, axis=0)

    return saved_metrics


def process_featureattribution(batch, **kwargs):
    saved_metrics = {}
    shap_vals = []
    raw_idxs = []
    for x in batch:
        cf = x[0]
        loc = x[1]

        shap_vals.append(cf)
        raw_idxs.append(loc)
    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()
    shap_vals = np.stack(shap_vals).squeeze()

    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    batch_avg_confidence = np.mean(orig_confidence.detach().numpy(), axis=0)
    batch_std_confidence = np.std(orig_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)

    batch_mean_along_timepoints = np.mean(np.mean(shap_vals, axis=0), axis=1)
    batch_mean_along_features = np.mean(np.mean(shap_vals, axis=0), axis=0)
    batch_avg_std_along_timepoints = np.mean(np.std(shap_vals, axis=2), axis=0)
    batch_avg_std_along_features = np.mean(np.std(shap_vals, axis=1), axis=0)

    batch_median_along_timepoints = np.mean(np.median(shap_vals, axis=0), axis=1)
    batch_median_along_features = np.mean(np.median(shap_vals, axis=0), axis=0)

    saved_metrics['batch_avg_confidence'] = batch_avg_confidence
    saved_metrics['batch_std_confidence'] = batch_std_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_mean_along_timepoints'] = batch_mean_along_timepoints
    saved_metrics['batch_mean_along_features'] = batch_mean_along_features
    saved_metrics['batch_avg_std_along_timepoints'] = batch_avg_std_along_timepoints
    saved_metrics['batch_avg_std_along_features'] = batch_avg_std_along_features
    saved_metrics['batch_median_along_timepoints'] = batch_median_along_timepoints
    saved_metrics['batch_median_along_features'] = batch_median_along_features

    return saved_metrics

def process_batch_comte(batch, **kwargs):
    return process_counterfactual(batch, **kwargs)

def process_batch_gradcam(batch, **kwargs):
    return process_featureattribution(batch, **kwargs)

def process_batch_nuncaf(batch, **kwargs):
    return process_counterfactual(batch, **kwargs)


def process_batch_timeshap(batch, **kwargs):
    saved_metrics = {}
    feature_shap_vals = []
    timepoint_shap_vals = []
    raw_idxs = []
    for x in batch:
        cf = x[0]
        loc = x[1]

        feature_shap_vals.append(cf[1].to_numpy()[:,3])
        timepoint_shap_vals.append(cf[0].to_numpy()[:,3])
        raw_idxs.append(loc)
    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()

    # metrics:
    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    batch_avg_confidence = np.mean(orig_confidence.detach().numpy(), axis=0)
    batch_std_confidence = np.std(orig_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)

    batch_mean_along_timepoints = np.mean(timepoint_shap_vals, axis=0)
    batch_mean_along_features = np.mean(feature_shap_vals, axis=0)

    try:
        batch_avg_std_along_timepoints = np.std(np.stack(timepoint_shap_vals), axis=0)
    except  TypeError as e:
        batch_avg_std_along_timepoints = 0
    try:
        batch_avg_std_along_features = np.std(feature_shap_vals, axis=0)
    except TypeError as e:
        batch_avg_std_along_features = 0

    batch_median_along_timepoints = np.median(timepoint_shap_vals, axis=0)
    batch_median_along_features = np.median(feature_shap_vals, axis=0)

    saved_metrics['batch_avg_confidence'] = batch_avg_confidence
    saved_metrics['batch_std_confidence'] = batch_std_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_mean_along_timepoints'] = batch_mean_along_timepoints
    saved_metrics['batch_mean_along_features'] = batch_mean_along_features
    saved_metrics['batch_avg_std_along_timepoints'] = batch_avg_std_along_timepoints
    saved_metrics['batch_avg_std_along_features'] = batch_avg_std_along_features
    saved_metrics['batch_median_along_timepoints'] = batch_median_along_timepoints
    saved_metrics['batch_median_along_features'] = batch_median_along_features

    return saved_metrics


def process_batch_windowshap(batch, **kwargs):
    return process_featureattribution(batch, **kwargs)


def reshape_to_examples(batch):
    return np.reshape(batch, (-1, 70, 15))

PROCESSES = {'comte': [process_batch_comte],
             'gradcam': [process_batch_gradcam],
             'nuncaf': [process_batch_nuncaf],
             'time_shap': [process_batch_timeshap],
             'window_shap': [process_batch_windowshap] }

PREPROCESSES = {'comte': [None],
             'gradcam': [None],
             'nuncaf': [reshape_to_examples],
             'time_shap': [None],
             'window_shap': [None] }


def load_analysis_files(directory, item_categories=[]):
    results_dict = {x: [] for x in item_categories}

    for cat in item_categories:
        list_of_pickels = os.listdir(directory + cat)
        for foldername in list_of_pickels:
            temp_path = directory + cat + "/" + foldername
            temp_files = os.listdir(temp_path)
            for file in temp_files:
                if file[-4:] == ".pkl":
                    temp_f = open(temp_path + "/" + file, "rb")
                    res_objects = pickle.load(temp_f)
                    temp_f.close()
                    results_dict[cat].append(res_objects)
    return results_dict


def process_perBatch_results(per_batch_data):
    final_res = {}
    for catg, batches_results in per_batch_data.items():
        keys_to_avg = batches_results[0].keys()
        avg_dict = {x:None for x in keys_to_avg}
        for item in batches_results:
            for metric_name, metric_val in item.items():
                if avg_dict[metric_name] is None:
                    avg_dict[metric_name] = metric_val/len(batches_results)
                else:
                    avg_dict[metric_name] += metric_val/len(batches_results)
        final_res[catg] = avg_dict

    return final_res

def plot_original_overlap_counterfactual(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3, min_max_dict=None, display_names=None, timepoints=None):    
    #Calculate the per-feature original-counterfactual differences
    
    total_n_plots = test_item.shape[-1]
    plottable_indexes = list(range(total_n_plots))
    extra_row = 0
    num_subs_to_remove = 0
    if total_n_plots % n_plots_horiz != 0:
        extra_row += 1
        num_subs_to_remove = n_plots_horiz - (total_n_plots % n_plots_horiz) 
    n_plots_vert = (total_n_plots // n_plots_horiz) + extra_row
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 3*total_n_plots), layout='constrained')
    
    #Create the time tick labels
    ts_len = explan_res[0].shape[1]
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 1 == 0:
            time_tick_lbls[j] = str(j)
            
    #Generate the plots
    for plot_num, i in enumerate(plottable_indexes):
        fn = feature_names[i]
        d_name = display_names[fn]
        
        #Only plot if counterfactual record exists
        if True:
            minv, maxv = min_max_dict[fn]

            if n_plots_horiz > 1:
                main_axis = axis[plot_num // n_plots_horiz, plot_num % n_plots_horiz]
            else:
                main_axis = axis[plot_num // n_plots_horiz]
                
            if fn == 'avpu':
                minv, maxv = -0.5,3.5
                main_axis.set_yticks([0,1,2,3])
                main_axis.set_yticklabels(['Alert', 'Responds to Voice', 'Responds to Pain', 'Unresponsive'])
                
            main_axis.plot(timepoints, test_item[:, :, i].flatten(), color='b', label='Feature values')
            main_axis.set_title(f"{d_name}")
            main_axis.set_xticks(timepoints, labels=time_tick_lbls)
            main_axis.set_ylabel("Feature values", color='b')
            main_axis.set_ylim([minv, maxv])
            main_axis.tick_params(axis='y', colors='b')

            overlay_plot = main_axis.twinx()
            
            if fn == 'avpu':
                minv, maxv = -0.5,3.5
                overlay_plot.set_yticks([0,1,2,3])
                overlay_plot.set_yticklabels(['Alert', 'Responds to Voice', 'Responds to Pain', 'Unresponsive'])
            
            overlay_plot.plot(timepoints, explan_res[0][:, :, i].flatten(), color='r', label='Counterfactual', linestyle='dashed')
            overlay_plot.set_ylabel("Counterfactual", color='r')
            overlay_plot.set_ylim([minv, maxv])
            overlay_plot.tick_params(axis='y', colors='r')
            
            

    image_loc = f"{explanation_output_folder}/{image_name_prefix}allFeatures.png" 
    print(f"Saving figure to {image_loc}")
    figure.savefig(image_loc)
    plt.close()


def plot_original_line_with_vals(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3, min_max_dict=None, display_names=None, timepoints=None):
    maxves = explan_res.max() + (0.1 * explan_res.max())
    minves = explan_res.min() - (0.1 * explan_res.max())

    explan_res = explan_res[0, :test_item.shape[1], :]
    
    plottable_indexes = range(len(feature_names))
    total_n_plots = len(feature_names)
    
    ordered_f_idxs = np.argsort(np.absolute(explan_res).sum(axis=0))[::-1]
         
    extra_row = 0
    num_subs_to_remove = 0
    if total_n_plots % n_plots_horiz != 0:
        extra_row += 1
        num_subs_to_remove = n_plots_horiz - (total_n_plots % n_plots_horiz) 
    n_plots_vert = (total_n_plots // n_plots_horiz) + extra_row
    
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 3*total_n_plots), layout='constrained')    
    
    ts_len = explan_res.shape[0]
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 1 == 0:
            time_tick_lbls[j] = str(j)
            
    #Generate the plots
    for plot_num, f_loc in enumerate(ordered_f_idxs):
        feature_name = feature_names[f_loc]
        d_name = display_names[feature_name]
     
        if n_plots_horiz > 1:
            working_subplot = axis[plot_num // n_plots_horiz, plot_num % n_plots_horiz]
        else:
            working_subplot = axis[plot_num // n_plots_horiz]
            
        minv, maxv = min_max_dict[feature_name]
        if feature_name == 'avpu':
            minv, maxv = -0.5,3.5
            working_subplot.set_yticks([0,1,2,3])
            working_subplot.set_yticklabels(['Alert', 'Responds to Voice', 'Responds to Pain', 'Unresponsive'])
            
        working_subplot.set_title(f"{d_name}")
        working_subplot.set_xticks(timepoints, labels=time_tick_lbls)

        working_subplot.plot(timepoints, test_item[:, :, f_loc].flatten(), color='b', label='Feature Value')
        working_subplot.set_ylabel("Feature values", color='b')
        working_subplot.tick_params(axis='y', colors='b')
        working_subplot.set_ylim([minv, maxv])

        overlay_plot = working_subplot.twinx()
        overlay_plot.bar(timepoints, explan_res[:, f_loc].flatten(), color='r', label='Importance values')
        overlay_plot.set_ylabel("Importance values", color='r')
        overlay_plot.tick_params(axis='y', colors='r')
        overlay_plot.set_ylim([minves, maxves])
        
        overlay_plot.axhline(y=0, color='red', linestyle=":", alpha=0.25)

        working_subplot.set_zorder(overlay_plot.get_zorder() + 1)
        working_subplot.patch.set_visible(False)
        
    image_loc = f"{explanation_output_folder}/{image_name_prefix}allFeatures.png"
    print(f"Saving figure to {image_loc}")
    figure.savefig(image_loc)
    plt.close()
    
    
def plot_original_line_with_dots(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3, min_max_dict=None):
    n_plots_vert = (len(feature_names) // n_plots_horiz) + 1
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(5, 3*len(feature_names)), layout='constrained')
    
    ts_len = test_item.shape[1]
    
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 5 == 0:
            time_tick_lbls[j] = str(j)
            
    for i, f in enumerate(feature_names):
        minv, maxv = min_max_dict[f]
        print(f"Anchors feat: {f} \t MINV: {minv} and MAXV: {maxv}")
        feature_name = feature_names[i]
        if n_plots_horiz > 1:
            working_subplot = axis[i // n_plots_horiz, i % n_plots_horiz]
        else:
            working_subplot = axis[i // n_plots_horiz]
        working_subplot.set_title(f"{feature_name}")
        working_subplot.set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)

        working_subplot.plot(list(range(0, ts_len)), test_item[:, :, i].flatten(), color='b', label='Original Instance')
        working_subplot.set_ylabel("Original Instance", color='b')
        working_subplot.tick_params(axis='y', colors='b')
        working_subplot.set_ylim([minv, maxv])

        overlay_plot = working_subplot.twinx()
        overlay_plot.plot(list(range(0, ts_len)), explan_res[:, i], 'o', color='r', label='Comparison Values', markersize=5)
        overlay_plot.set_ylabel("Comparison Values", color='r')
        overlay_plot.tick_params(axis='y', colors='r')
        overlay_plot.set_ylim([minv, maxv])

        working_subplot.set_zorder(overlay_plot.get_zorder() + 1)
        working_subplot.patch.set_visible(False)

    
    image_loc = f"{explanation_output_folder}/{image_name_prefix}allFeatures.png"
    print(f"Saving figure to {image_loc}")
    figure.savefig(image_loc)
    plt.close()



def run_analysis():
    directory = '_saved_models/LSTM_mine/'
    item_categories = ['Anchors', 'COMTE', 'GradCAM', 'NUNCF', 'WindowSHAP']
    replication_size = 10

    data = load_analysis_files(directory, item_categories)

    ctg_batch_res = {x:[] for x in item_categories}
    for catg in item_categories:
        catg_data = data[catg]

        for i in range(0, len(catg_data)//replication_size):
            start = i * replication_size
            end = (i+1) * replication_size
            same_sample_batch = catg_data[start:end]

            proc_fn = PROCESSES[catg][0]
            pre_fn = PREPROCESSES[catg][0]
            batch_results = proc_fn(same_sample_batch, model=model, orig_inputs=orig_inputs, preprocess=pre_fn)
            ctg_batch_res[catg].append(batch_results)

    all_res = process_perBatch_results(ctg_batch_res)


def cf_histogram(exp_names, float_tolerance=0.0):
    data = {}
    for exp_name in exp_names:
        foldpath = f"_saved_models/{exp_name}/"
        filepath = foldpath + "all_explanations.pkl"
        f = open(filepath, "rb")
        all_explanations = pickle.load(f)
        f.close()
        exp_results = all_explanations['CoMTE']['result_store']
        comte_exps = exp_results['explanations']
        comte_orig = exp_results['samples_explained']

        exp_feat_n = exp_name.split("_feat")[1][0]

        for i, orig in enumerate(comte_orig):
            orig_x = orig['x']
            exp_x = comte_exps[i][0]

            if exp_feat_n in data.keys():
                data[exp_feat_n].append((orig_x, exp_x))
            else:
                data[exp_feat_n] = [(orig_x, exp_x)]

    histogram_data = {}

    for feat_n, list_tups in data.items():
        for tup in list_tups:
            itm = tup[0]
            exp = tup[1]
            diff = np.sum(exp - itm, axis=1).squeeze()
            diff_loc = (diff > float_tolerance) + (diff < -float_tolerance)
            sum_diffs = sum(diff_loc)

            if sum_diffs in histogram_data.keys():
                histogram_data[sum_diffs] += 1
            else:
                histogram_data[sum_diffs] = 1

    t_l = list(range(1, len(exp_names)+1))
    h = bar(height=list(histogram_data.values()), x=t_l, tick_label=t_l )
    for i in range(0,5):
        plt.text(i+1, list(histogram_data.values())[i], list(histogram_data.values())[i])

    temp = {int(k):len(v) for k,v in data.items()}

if __name__ == "__main__":
    e_list = ["Exp_COMTE_conf99_feat1", "Exp_COMTE_conf99_feat2", "Exp_COMTE_conf99_feat3", "Exp_COMTE_conf99_feat4", "Exp_COMTE_conf99_feat5"]
    cf_histogram(e_list)
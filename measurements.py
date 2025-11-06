import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import copy
import scipy


def get_byquestion_data(response_data, fields_to_remove):
    byQ_data = {}
    for participant_num in range(len(response_data)):
        participant_data = response_data.iloc[participant_num]
        participant_cols_w_data = participant_data.dropna()
        pid = participant_data['record_id']
        byQ_data[pid] = {}
        
        for q_n in ['_q1', '_q2', '_q3', '_q4', '_q5']:
            for q_t in ['init', 'first_', 'second_', 'third_']:
                q_data = []
                # record_id = participant_cols_w_data['record_id']
            
                q_ranks = get_question_rankings(byQ_data=participant_cols_w_data, qtype=q_t, qnum=q_n, excludes=fields_to_remove)
                # if len(init_q1) != 5:
                #     print(f"WARNING Record:{record_id} for {q_n} {q_t} does not have 5 items selected: {len(init_q1)}")
                # q_data.append(init_q1)
                new_qr={}
                for k,v in q_ranks.items():
                    new_qr[v] = k.replace(q_n,"").replace('initial_',"").replace(q_t,"").replace('_heartrate',"heartrate")
                
                if q_n.replace("_","") not in byQ_data[pid]:
                    byQ_data[pid][q_n.replace("_","")] = {q_t.replace("_",""): new_qr}
                else:
                    byQ_data[pid][q_n.replace("_","")][q_t.replace("_","")] = new_qr
                
            # z = get_answer_distrib(list_of_answer_dicts=q_data, topcount=5)
            # print("\n".join([f"\t{k}:{v}" for k,v in z.items()]))
            # print("--------------------------------------")
        # res[q_n.replace("_", "")] = z
    return byQ_data


def get_question_rankings(byQ_data, qtype='init', qnum='q1', excludes=[]):
    records = {}
    for col in byQ_data.index:
        if col.startswith(qtype) and qnum in col and len([x for x in excludes if x in col])==0:
            # print(col)
            records[col] = byQ_data[col]

    return dict(sorted(records.items(), key=lambda item: item[1], reverse=True))

    
def make_per_question_data(responsedata, byQ_data, q_time_mapping):
    per_question_data = []
    headers = None
    
    for record_id, rec_data in byQ_data.items():
        holder_list = []
        
        for qnum, qdata in rec_data.items():
            for ifst, idata in qdata.items():
                matching = responsedata[responsedata['record_id'] == record_id]
                controlerid = int(matching[f'{qnum}_order_control_int'])
                mapped_val = q_time_mapping[controlerid][ifst]
                
                holder = {'record_id': record_id}
                holder['question_number'] = qnum
                holder['question_order'] = ifst
                holder['control_val'] = controlerid
                holder['exp_type'] = mapped_val
                holder['is_nurse_or_not'] = matching['is_nurse_or_not'].item()
                holder['site'] = matching['site'].item()
                holder['post_attrib_longshort'] = matching['post_attrib_longshort'].item()
                holder['post_rb_length'] = matching['post_rb_length'].item()
                
                holder['RRT'] = 'Yes' if matching['is_rrt'].item() == 1 else 'No'

                if (matching['is_nurse'] == 1).any():
                     holder['role'] = 'Nurse'
                elif (matching['is_trainee'] == 1).any():
                     holder['role'] = 'Physician Trainee'
                elif (matching['is_attending'] == 1).any():
                     holder['role'] = 'Attending Physician'
                elif (matching['is_midlevel'] == 1).any():
                     holder['role'] = 'Mid-level Provider'


                holder['initial_r_expimportance'] = matching['initial_r_expimportance'].item()
                holder['initial_r_mltrust'] = matching['initial_r_mltrust'].item()
                holder['initial_t_whatexpshouldbe'] = matching['initial_t_whatexpshouldbe'].item()
                holder['demo_gender'] = matching['demo_gender'].item()
                holder['demo_age_range'] = matching['demo_age_range'].item()
                holder['unified'] = matching['unified'].item()
    
                most_to_least = [idata[i] if i in idata else np.nan for i in range(1,6) ]
                holder['most_to_least_important'] = most_to_least
                
                unranked_pos_count = len([z for z in most_to_least if z != np.nan])
                holder['unranked_pos_count'] = unranked_pos_count
    
                if ifst != 'init':
                    previous_q = holder_list[-1]['most_to_least_important']
                    holder['previous_most_to_least'] = previous_q
    
                    same_idx_as_prev = []
                    same_idx_as_prev_count = 0
                    not_in_prev = []
                    not_in_prev_count = 0
                    any_dif_from_prev = 0
                    shifted_from_prev = []
                    shifted_from_prev_count = 0
                    shifted_from_prev_dist = 0
                    shifted_from_prev_dist_wm = 0
    
                    
                    for x in most_to_least:
                        if x in previous_q:
                            if most_to_least.index(x) == previous_q.index(x):
                                same_idx_as_prev.append(x)
                                same_idx_as_prev_count += 1
                            else:
                                shifted_from_prev.append([previous_q.index(x), most_to_least.index(x)])
                                shifted_from_prev_count += 1
                                shifted_from_prev_dist += abs(most_to_least.index(x) - previous_q.index(x))
                                shifted_from_prev_dist_wm += abs(most_to_least.index(x) - previous_q.index(x))
                                any_dif_from_prev += 1
                        else:
                            not_in_prev.append(x)
                            not_in_prev_count += 1 
                            any_dif_from_prev += 1
                            shifted_from_prev_dist_wm += abs(5 - most_to_least.index(x))
                            
    
                    holder['same_idx_as_prev'] = same_idx_as_prev
                    holder['same_idx_as_prev_count'] = same_idx_as_prev_count
                    holder['not_in_prev'] = not_in_prev
                    holder['not_in_prev_count'] = not_in_prev_count
                    holder['any_dif_from_prev'] = any_dif_from_prev
                    holder['shifted_from_prev'] = shifted_from_prev
                    holder['shifted_from_prev_count'] = shifted_from_prev_count
                    holder['shifted_from_prev_dist'] = shifted_from_prev_dist
                    holder['shifted_from_prev_dist_wm'] = shifted_from_prev_dist_wm

                    # print(f"most_to_least: {most_to_least}")
                    count_changes_from_prev = len([ii for ii in range(0,5) if most_to_least[ii] != previous_q[ii]])
                    count_same_from_prev = len([ii for ii in range(0,5) if most_to_least[ii] == previous_q[ii]])
    
                    holder['count_changes_from_prev'] = count_changes_from_prev
                    holder['count_same_from_prev'] = count_same_from_prev
    
                    holder['understand_exp'] = matching[f"understandexp_{ifst}_{qnum}"].item()
                    # print(f"understand_exp: {holder['understand_exp'] }")
                    holder['increase_trust_ml_model'] = matching[f"increasetrust_{ifst}_{qnum}"].item()
                    holder['increase_understand_ml_model'] = matching[f"increaseunderstand_{ifst}_{qnum}"].item()
    
                else: #for init case
                    holder['previous_most_to_least'] = []
                    holder['same_idx_as_prev'] = []
                    holder['same_idx_as_prev_count'] = np.nan
                    holder['not_in_prev'] = []
                    holder['not_in_prev_count'] = np.nan
                    holder['any_dif_from_prev'] = np.nan
                    holder['shifted_from_quesprev'] = []
                    holder['shifted_from_prev_count'] = np.nan
                    holder['shifted_from_prev_dist'] = np.nan
                    holder['shifted_from_prev_dist_wm'] = np.nan
                    holder['count_changes_from_prev'] = np.nan
                    holder['count_same_from_prev'] = np.nan
                    holder['understand_exp'] = np.nan
                    holder['increase_trust_ml_model'] = np.nan
                    holder['increase_understand_ml_model'] = np.nan
    
                holder_list.append(holder)
        per_question_data += holder_list
        headers = holder_list[0].keys()
                
    return pd.DataFrame(per_question_data, columns=headers)


def add_vertical_noise(series, max_noise=0.05):
    return series + np.random.uniform(-max_noise, max_noise, size=len(series))


def levenshtein_distance_lists(list1, list2):
    #Calculates the Levenshtein distance between two lists.
    n = len(list1)
    m = len(list2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if list1[i - 1] == list2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[n][m]

def swap_distance(list1, list2):
    dist = 0
    uniques =  set(list1).union(set(list2))
    for x in uniques:
        if x not in list1:
            dist += 1
        # if x not in list2:
        #     dist += 1
    # print(f"list1: {list1}, list2: {list2}, dist={dist}")
    return dist


def process_change_fields(pq_data, data, postfields, color_on_field_1, color_on_field_2):
    recs = {}
    for r in range(len(pq_data)):
        record_q_data = pq_data.iloc[r]
        record_id = record_q_data['record_id']
    
        if record_id not in recs:
            recs[record_id] = {'ue_at':[], 'ue_cf':[], 'ue_rb':[],'it_at':[], 'it_cf':[], 'it_rb':[],'iu_at':[], 'iu_cf':[], 'iu_rb':[],'in_at':[], 'in_cf':[], 'in_rb':[]}
        matching = data[data['record_id'] == record_id]
        
        #invert the mappings so that the lower rankings (i.e. 1st place) correspond to higher values
        for f in postfields:
            recs[record_id][f] = matching[f].item()
    
        #manually define the 
        if record_q_data['exp_type'] == 'Attribution':
            recs[record_id]['ue_at'].append(record_q_data['understand_exp'])
            recs[record_id]['it_at'].append(record_q_data['increase_trust_ml_model'])
            recs[record_id]['iu_at'].append(record_q_data['increase_understand_ml_model'])
            recs[record_id]['in_at'].append(record_q_data['count_changes_from_prev'])
        if record_q_data['exp_type'] == 'Counterfactual':
            recs[record_id]['ue_cf'].append(record_q_data['understand_exp'])
            recs[record_id]['it_cf'].append(record_q_data['increase_trust_ml_model'])
            recs[record_id]['iu_cf'].append(record_q_data['increase_understand_ml_model'])
            recs[record_id]['in_cf'].append(record_q_data['count_changes_from_prev'])
        if record_q_data['exp_type'] == 'Rule-based':
            recs[record_id]['ue_rb'].append(record_q_data['understand_exp'])
            recs[record_id]['it_rb'].append(record_q_data['increase_trust_ml_model'])
            recs[record_id]['iu_rb'].append(record_q_data['increase_understand_ml_model'])
            recs[record_id]['in_rb'].append(record_q_data['count_changes_from_prev'])
            
        recs[record_id]['record_id'] = record_id
        recs[record_id][color_on_field_1] = 'Nurse' if matching['is_nurse'].item() == 1 else 'Physician'
        recs[record_id][color_on_field_2] = matching[color_on_field_2].item()
    
    df_scatter = pd.DataFrame.from_dict(recs, orient='index')
    return df_scatter
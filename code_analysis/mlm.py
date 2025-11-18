import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import copy


def create_mixed_effects_model(df, fixed_cols, random_cols, target):
    df_model = df.copy()

    #  Define the data based on the cols
    required_cols = fixed_cols + random_cols + [target]

    # Remove rows with missing values in required cols
    df_model = df_model.dropna(subset=required_cols)

    if len(df_model) == 0:
        raise ValueError("No valid rows remaining after removing missing values")

    # Fixed effects formula string for statsmodel
    fixed_formula = f"{target} ~ " + " + ".join(fixed_cols)
    print(f"Formula is: {fixed_formula}")

    if len(random_cols) == 1:
        print(f"Fitting mixed effects model with random intercept for: {random_cols[0]}")
        model = mixedlm(fixed_formula, df_model, groups=df_model[random_cols[0]])
        model_result = model.fit()
        return model, model_result, fixed_formula, df_model
    else:
        raise NotImplementedError()

    return model_results, df_model


def make_post_mlm(local_data, cols_to_flatten, fixed_effect_cols, random_effect_cols):
    new_data = []
    for i, row in local_data.iterrows():
        row_as_dict = row.to_dict()
        for v in cols_to_flatten:
            del row_as_dict[v]
    
        for v in cols_to_flatten: 
            # print(f"v is: {v}")

            dup_row = copy.deepcopy(row_as_dict)

            if v.startswith('post_at'):
                exp_type = 'Attribution'
            elif v.startswith('post_cf'):
                exp_type = 'Counterfactual'
            elif v.startswith('post_rb'):
                exp_type = 'Rule-based'
            else:
                raise NotImplementedError()
    
            dup_row['exp_type'] = exp_type
            dup_row['target_value'] = row[v]
            new_data.append(dup_row)
            # print(row_as_dict)
    new_data = pd.DataFrame(new_data)
    
    # print(f"{post_changeunderstand_header.upper()}")
    target_col = 'target_value'
    
    temp_data = new_data[~new_data[target_col].isna()]
    temp_data = temp_data[fixed_effect_cols + random_effect_cols + [target_col]]

    # print(temp_data['exp_type'])
    lmm_model_7, lmm_result_7, lmm_fixed_formula_7, model_data_7 = create_mixed_effects_model(
        df=temp_data,
        fixed_cols=fixed_effect_cols,
        random_cols=random_effect_cols,
        target=target_col
    )
    
    print(lmm_result_7.summary())
    fixed_effect_cols
    return new_data
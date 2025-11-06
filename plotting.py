import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import copy
import scipy

from measurements import add_vertical_noise

likert_start = -2
likert_stop = 2 + 1

HUE_COLORS = {
    'Nurse':'orange', 'Physician':'blue',
    'UW-Madison': 'red', 'U-Chicago': 'green',
    'Clinicians': '#437bb6'}

def plot_trajectory(df, title, xname, yname, means, varname='question_number', val='understand_exp', 
                    ytick=[1, 2, 3, 4, 5],
                    yticklbl=None,
                    xtick=None,
                    xticklbl=None):    
    melted_data = df
    sns.set_style("white")

    plt.figure(figsize=(10, 4))
    many_colors_palette = sns.color_palette('rocket', n_colors=len(melted_data['record_id'].unique()))

    ax = sns.lineplot(x=varname, y=val, data=melted_data, style='record_id', 
                      color='blue', # palette=many_colors_palette, hue='record_id',
                      alpha=0.15, dashes=False, marker='>')
    ax.set_yticks(ytick)
    ax.set_yticklabels(yticklbl)
    if xtick:
        ax.set_xticks(xtick)
    if xticklbl:
        ax.set_xticklabels(xticklbl)

    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.grid(True)

    for i,m in enumerate(means):
        if i==0:
            ival = xtick[i] if xtick else i 
            plt.scatter(ival, m, color='red', marker='o', s=50, label='Mean')
        else:
            ival = xtick[i] if xtick else i 
            plt.scatter(ival, m, color='red', marker='o', s=50)


    handles, labels = ax.get_legend_handles_labels()
    new_handles = [h for i, h in enumerate(handles)][-2:]
    new_labels = ['Single Participant'] + [l for i, l in enumerate(labels)][-1:]
    plt.legend(new_handles, new_labels, title='Legend', ncol=3, bbox_to_anchor=(1, 1))


def display_trajectories(perQ_data, traj_data, question_col_name, agreelist, titlepart, finalpart):
    for modifier in ['Rule-based', 'Attribution', 'Counterfactual']:
        # print("--------------------------------------------------------------")
        if modifier != 'ALL':
            mod_text = modifier
            temp_data = perQ_data[perQ_data['exp_type'] == modifier]
            # print(temp_data)
        else:
            mod_text = "For Each Case"
            temp_data = perQ_data
        
        temp_data = temp_data[~pd.isna(perQ_data[question_col_name])]
        # print(temp_data)
    
        if modifier != 'ALL':
            temp_data.insert(2, 'qno', temp_data['question_number'])
            xt = list(range(0,len(temp_data['qno'].unique())+1)) 
        else:
            temp_data.insert(2, 'qno', temp_data['question_number'] + '_' + temp_data['question_order'])
            # print(temp_data)
            xt = ['Q1_first', 'Q2_first', 'Q3_first', 'Q4_first', 'Q5_first', 'Final']
            # xt = [0 ,3, 6, 9, 12,13]
            # print(f"xt: {xt}")
            
        temp_data = temp_data[['record_id', question_col_name, 'qno']]
        # print(temp_data)
        # print(xt)
        
        tfin = pd.concat([traj_data['record_id'], traj_data[[f'post_at_{finalpart}',f'post_cf_{finalpart}',f'post_rb_{finalpart}']].mean(axis=1)], axis=1)
        # print(f"tfin: {tfin.shape}")
        tfin = tfin.reset_index().rename({0:question_col_name}, axis=1)
        # tfin = tfin.drop(['index'], axis=1)
        tfin['qno'] = 'Final'
        # print(f"tfin: {tfin}")
        temp_data = pd.concat([temp_data, tfin], axis=0, ignore_index=True)
        temp_data[[question_col_name]] = temp_data[[question_col_name]]

        # print(f"temp_data 'record_id': {temp_data['record_id'].unique()} , tfin:{tfin.index}")

        # tmeans = temp_data[question_col_name]
        # print(f"tmeans: {tmeans}")
        tmeans = [x for x in perQ_data[['question_number', question_col_name]].groupby('question_number').mean()[question_col_name].tolist()]
        # tmeans.append(tfin.loc[[x for x in tfin.index if 'post' in x]].mean().item())
        tmeans.append(tfin[question_col_name].mean().item())
        
        plot_trajectory(df=temp_data, 
                        title=f"({mod_text}) {titlepart}", 
                        xname='Case Number', 
                        yname='', 
                        varname='qno',
                        val=question_col_name,
                        means=tmeans,
                        yticklbl=agreelist[::-1],
                        xtick=xt,
                        xticklbl=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Final'])



def show_side_by_side_bars(score_col, grouptype, xlbl, data_used, ylbl='# Features w/ Ranking Changes', 
                           fintitle="Overall - Number of Changes in Ranking",
                          suptitle='Number of Changes in Feature Importance Ranking (from Previous Screen) After Seeing Explanation of This Type',
                          xtick=None, xticklbl=None, ytick=None, yticklbl=None, ylim_tuple=(0, 5),
                          selectcrit='exp_type', skipbyquestion=False, noise_val=0.05,
                          color_on=None, g1_override='Physician', g2_override='Nurse'):
    df = copy.deepcopy(data_used)
    df = df[df['question_order'] != 'init']
    df['question_number'] = df['question_number'].astype('category')


    if not skipbyquestion:

        temp_df = copy.deepcopy(df)
        temp_df[score_col] = add_vertical_noise(df[score_col], noise_val)
        g = sns.catplot(
            data=temp_df,
            x=grouptype,
            y=score_col,
            col='question_number',
            kind='strip',
            height=5, # Height of each mingraph
            # aspect=1,
            hue=color_on,
            palette=HUE_COLORS,
            jitter=noise_val*2,
            alpha=0.3
        )
        sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1.05))
        
        additional_labels = []
        additional_handles = []
        for category_name, ax in g.axes_dict.items():
            
            if color_on: 
                               
                mean_values = df.groupby([grouptype, 'question_number', color_on]).mean(numeric_only=True)
                mean_values = mean_values.reset_index()
                # print(f"mean_values: {mean_values}")

                for z in df[color_on].unique():
                    current_category_means = mean_values[np.logical_and(
                        mean_values['question_number'] == category_name,
                        mean_values[color_on] == z,
                        )]
                    
                    for i, row in current_category_means.iterrows():
                        group_name = row[selectcrit]
                        mean_value = row[score_col]
                        # print(f"group_name: {group_name} \t mean_value: {mean_value}")
                
                        x_pos = g.axes[0,0].get_xticks()[0] # Initialize with first tick position
                        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
                        x_pos = tick_labels.index(group_name)
                
                        line1 = ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_value, mean_value],
                                color=HUE_COLORS[z], #'red', 
                                linestyle='-', linewidth=2, zorder=3) # zorder to ensure it's on top
                        ax.margins(y=0.1)

                        # print(f"line1: {line1}")
                        additional_labels.append(line1[0].get_label())
                        additional_handles.append(line1[0])
            else:
                mean_values = df.groupby([grouptype, 'question_number']).mean(numeric_only=True)
                mean_values = mean_values.reset_index()
                current_category_means = mean_values[mean_values['question_number'] == category_name]
            
                for i, row in current_category_means.iterrows():
                    group_name = row[selectcrit]
                    mean_value = row[score_col]

                    # print(f"group_name: {group_name}, mean_value: {mean_value}")
            
                    x_pos = g.axes[0,0].get_xticks()[0] # Initialize with first tick position
                    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
                    x_pos = tick_labels.index(group_name)
            
                    line1 = ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_value, mean_value],
                            color='red', 
                            linestyle='-', linewidth=2, zorder=3) # zorder to ensure it's on top
                    ax.margins(y=0.1)

                    additional_labels.append(line1.get_label())
                    additional_handles.append(line1)
        # g.legend.set_title("Legend")
        catplot_handles, catplot_labels = g.legend.legend_handles, [text.get_text() for text in g.legend.get_texts()]
        all_handles = catplot_handles + additional_handles
        all_labels = catplot_labels + additional_labels
        g.legend.set_visible(False)
        plt.legend(handles=[all_handles[i] for i in [0,1,2,5]], labels=[all_labels[i] for i in [0,1]]+[f'{g1_override} Mean', f'{g2_override} Mean'], title="Legend",
           bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
        # Set titles and labels
        g.set_axis_labels(xlbl, ylbl)
        g.set_titles("{col_name}")
        if ytick and yticklbl:
            print(f"test1")
            g.set(ylim=ylim_tuple, yticks=ytick, yticklabels=yticklbl)
        else:
             print(f"test2")
             g.set(ylim=ylim_tuple)
        plt.suptitle(suptitle, y=1.02) # Overall title for the figure

        g.set_yticklabels(yticklbl)

        plt.tight_layout()
        plt.show()

    overall_avg_scores = copy.deepcopy(data_used)
    overall_avg_scores = overall_avg_scores[overall_avg_scores['question_order'] != 'init']#.groupby([grouptype])#.mean(numeric_only=True)

    temp_df = copy.deepcopy(overall_avg_scores)
    temp_df[score_col] = add_vertical_noise(df[score_col], noise_val)

    ax_overall = sns.stripplot(data=temp_df, 
                               x=grouptype, 
                               y=score_col, 
                               hue=color_on,
                               palette=HUE_COLORS,
                               alpha=0.4)
    ax_overall.set_title(fintitle)
    ax_overall.set_xlabel(xlbl)
    ax_overall.set_ylabel(ylbl)
    # print(f"test3")
    ax_overall.set_ylim(ylim_tuple)

    
    x_tick_labels = [label.get_text() for label in ax_overall.get_xticklabels()]
    
    if color_on:
        # print(f"overall_avg_scores: {overall_avg_scores.columns}")
        temp = overall_avg_scores.groupby([selectcrit, color_on]).mean(numeric_only=True)[score_col].reset_index()
        # print(f"temp: {temp}")

        for zz in overall_avg_scores[color_on].unique():
            # print(f"zz: {zz}")
            for index, row in temp.iterrows():
                group = row[selectcrit]
                mean_val = row[score_col]
            
                x_pos = x_tick_labels.index(group)
                row_zz = row[color_on]
            
                # Draw a short horizontal line at the for this group
                ax_overall.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val],
                        color=HUE_COLORS[row_zz], linestyle='-', linewidth=2, zorder=3,
                        label=f'{group}') # Label for legend
    else:
        temp = overall_avg_scores.groupby(selectcrit).mean(numeric_only=True)[score_col].reset_index()
        for index, row in temp.iterrows():
            group = row[selectcrit]
            mean_val = row[score_col]
        
            x_pos = x_tick_labels.index(group)
        
            # Draw a short horizontal line at the for this group
            ax_overall.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val],
                    color='red', linestyle='-', linewidth=2, zorder=3,
                    label=f'{group}') # Label for legend
            
        
    handles, labels = ax_overall.get_legend_handles_labels()
    
    new_handles = handles[:2] +  [handles[3]] +  [handles[2]] #+ additional_handles#[[h for i, h in enumerate(handles)][-1]]
    new_labels = labels[:2] + [f'{g1_override} Mean', f'{g2_override} Mean'] #+ additional_labels#['Mean']
    # print(f"new_handles: {new_handles} \t new_labels:{new_labels}")
    
    ax_overall.legend(new_handles, new_labels, title='Legend', ncol=1, bbox_to_anchor=(1, 1))

    # g.set(yticks=ytick)
    
    # for ax in g.axes.flat:
    #     # Modify y-axis tick labels for each subplot
    #      g.set(yticks=ytick)
    #      ax.set_yticklabels(yticklbl)

    if ytick:
        ax_overall.set_yticks(ytick)
    #     g.set(yticks=ytick)
    #     print(f"ytick: {ytick}")
    if yticklbl:
    #     print(f"yticklbl: {yticklbl}")
        ax_overall.set_yticklabels(yticklbl)
    #     g.set_yticklabels(yticklbl)
    if xtick:
        ax_overall.set_xticks(xtick)
    #     g.set(xticks=xtick)
    if xticklbl:
        ax_overall.set_xticklabels(xticklbl)
    #     g.set_xticklabels(xticklbl)


def pre_vs_post(response_data, agreelist, pre_var, post_var,
               suptitle = "It is important to see explanations of predictions for machine learning models",
               post_title="Post-survey",
               pre_title="Pre-survey",
               color_on=None):
    fig, axes = plt.subplots(1, 2, figsize=(4, 4), sharey=True) # 1 row, 2 columns, share y-axis

    if color_on:
        for xx in response_data[color_on].unique():
            subsetdata = response_data[response_data[color_on] == xx]
            colorxx = HUE_COLORS[xx]

            ########################################################################################
            # 4. Create the first plot  ############################################################
            if isinstance(pre_var, list):
                pre_x = (subsetdata[pre_var]).mean(axis=1)
            else:
                pre_x = (subsetdata[pre_var])
            pre_x = add_vertical_noise(pre_x, max_noise=0.05)

            sns.stripplot(y=pre_x, ax=axes[0], s=8, color=colorxx, alpha=0.4, label=xx)
            axes[0].set_title(pre_title)
            axes[0].set_xlabel('')
            axes[0].set_xticks([0])
            axes[0].set_xticklabels([''])

            axes[0].set_ylabel('')

            axes[0].axhline(pre_x.mean(),
                            color=colorxx, # Use a different color for distinction if preferred
                            linestyle='--',
                            linewidth=2,
                            label=f'{xx} Mean',
                            zorder=3)
            axes[0].legend(loc='lower right', fontsize='small') # Add legend to this subplot

            ####################################################################################
            #create the second plot ############################################################
            if isinstance(post_var, list):
                post_x = (subsetdata[post_var]).mean(axis=1)
            else:
                post_x = (subsetdata[post_var])
            post_x = add_vertical_noise(post_x, max_noise=0.05)

            sns.stripplot(y=post_x, ax=axes[1], s=8, color=colorxx, alpha=0.4, label=xx) # s is marker size
            axes[1].set_title(post_title)

            axes[1].set_xlabel('')
            axes[1].set_xticks([0])
            axes[1].set_xticklabels([''])

            axes[1].set_ylabel('Importance Value')
            axes[1].set_yticks(list(range(likert_start, likert_stop)))
            axes[1].set_yticklabels(agreelist[::-1])

            axes[1].axhline(post_x.mean(),
                            color=colorxx,
                            linestyle='--',
                            linewidth=2,
                            label=f'{xx} Mean', # Label for legend
                            zorder=3) # Ensure line is on top of  points
            axes[1].legend(loc='lower right', fontsize='small') # Add legend to this subplot
            

    else:
        if isinstance(post_var, list):
            post_x = (response_data[post_var]).mean(axis=1)
        else:
            post_x = (response_data[post_var])
    
        sns.stripplot(y=post_x, ax=axes[1], s=5) # s is marker size
        axes[1].set_title(post_title)
        axes[1].set_ylabel('Importance Value')
        axes[1].set_xlabel('')
        axes[1].set_xticks([])
        axes[1].set_yticks(list(range(likert_start, likert_stop)))
        axes[1].set_yticklabels(agreelist[::-1])
        axes[1].axhline(post_x.mean(),
                        color='red',
                        linestyle='--',
                        linewidth=2,
                        label=f'Mean', # Label for legend
                        zorder=3) # Ensure line is on top of  points
        axes[1].legend(loc='lower right', fontsize='small') # Add legend to this subplot
        
        
        # 4. Create the second  plot on the right axis
        if isinstance(pre_var, list):
            pre_x = (response_data[pre_var]).mean(axis=1)
        else:
            pre_x = (response_data[pre_var])
        sns.stripplot(y=pre_x, ax=axes[0], s=5)
        axes[0].set_title(pre_title)
        axes[0].set_ylabel('') # No y-label on the second plot because sharey=True
        axes[0].set_xlabel('') # Remove the default x-label if not needed
        axes[0].set_xticks([]) # Remove x-axis ticks as there's only one category
        axes[0].axhline(pre_x.mean(),
                        color='blue', # Use a different color for distinction if preferred
                        linestyle='--',
                        linewidth=2,
                        label=f'Mean',
                        zorder=3)
        axes[0].legend(loc='lower right', fontsize='small') # Add legend to this subplot
    
    
    # Optional: Add overall title and adjust layout
    plt.suptitle(f'Pre-survey vs. Post-survey : {suptitle}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout, leave space for suptitle
    
    plt.show()


def pre_vs_post_by_explanation(response_data, agreelist, pre_var, post_var, n_extra=3,
               suptitle = "It is important to see explanations of predictions for machine learning models",
               post_title="Post-survey",
               pre_title="Pre-survey",
               exp_set=[],
               color_on=None):
    fig, axes = plt.subplots(1, 1+n_extra, figsize=(10, 6), sharey=True) # 1 row, 2 columns, share y-axis

    if color_on:
        for co_val in response_data[color_on].unique():
            subsetdata = response_data[response_data[color_on] == co_val]
            if isinstance(pre_var, list):
                pre_x = (subsetdata[pre_var]).mean(axis=1)
            else:
                pre_x = (subsetdata[pre_var])
            pre_x = add_vertical_noise(pre_x, max_noise=0.05)
            colorxx = HUE_COLORS[co_val]

            g = sns.stripplot(y=pre_x, ax=axes[0], s=8, color=colorxx, alpha=0.3)
            g.legend()

            axes[0].set_title(pre_title)

            axes[0].set_xlabel('')
            axes[0].set_xticks([0])
            axes[0].set_xticklabels([''])
            axes[0].set_ylabel('')

            axes[0].axhline(pre_x.mean(),
                            color=colorxx,
                            linestyle='--',
                            linewidth=2,
                            label=f'{co_val} Mean',
                            zorder=3)
            axes[0].legend(loc='lower right', fontsize='small')


            for i in range(1, n_extra+1):
                exp_type = exp_set[i-1]
                single_post_var = post_var[i-1]
                
                post_x = (subsetdata[single_post_var])
                post_x = add_vertical_noise(post_x, max_noise=0.05)
                
                gi = sns.stripplot(y=post_x, ax=axes[i], s=8, color=colorxx, alpha=0.3) # s is marker size
                gi.legend()

                axes[i].set_title(f"{exp_type} {post_title}")
                axes[i].set_ylabel('Importance Value')
                axes[i].set_xlabel('')
                axes[i].set_xticks([0])
                axes[i].set_xticklabels([''])
                axes[i].set_ylabel('')

                axes[i].set_yticks(list(range(likert_start, likert_stop)))
                axes[i].set_yticklabels(agreelist[::-1])
                axes[i].axhline(post_x.mean(),
                                color=colorxx,
                                linestyle='--',
                                linewidth=2,
                                label=f'{co_val} Mean', # Label for legend
                                zorder=3) # Ensure line is on top of points
                axes[i].legend(loc='lower right', fontsize='small') # Add legend to this subplo

    else:
        if isinstance(pre_var, list):
            pre_x = (response_data[pre_var]).mean(axis=1)
        else:
            pre_x = (response_data[pre_var])

        sns.stripplot(y=pre_x, ax=axes[0], s=5)
        axes[0].set_title(pre_title)
        axes[0].set_ylabel('') # No y-label on the second plot because sharey=True
        axes[0].set_xlabel('') # Remove the default x-label if not needed
        axes[0].set_xticks([]) # Remove x-axis ticks as there's only one category
        axes[0].axhline(pre_x.mean(),
                        color='blue', # Use a different color for distinction if preferred
                        linestyle='--',
                        linewidth=2,
                        label=f'Mean',
                        zorder=3)
        axes[0].legend(loc='lower right', fontsize='small') # Add legend to this subplot


        for i in range(1, n_extra+1):
            exp_type = exp_set[i-1]
            single_post_var = post_var[i-1]
            
            post_x = (response_data[single_post_var])
            
            sns.stripplot(y=post_x, ax=axes[i], s=5) # s is marker size
            axes[i].set_title(f"{exp_type} {post_title}")
            axes[i].set_ylabel('Importance Value')
            axes[i].set_xlabel('')
            axes[i].set_xticks([])
            axes[i].set_yticks(list(range(likert_start, likert_stop)))
            axes[i].set_yticklabels(agreelist[::-1])
            axes[i].axhline(post_x.mean(),
                            color='red',
                            linestyle='--',
                            linewidth=2,
                            label=f'Mean', # Label for legend
                            zorder=3) # Ensure line is on top of points
            axes[i].legend(loc='lower right', fontsize='small') # Add legend to this subplo

    plt.suptitle(f'Pre-survey vs. Post-survey : {suptitle}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout, leave space for suptitle
    
    plt.show()


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str, title: str = "Scatter Plot of A vs B", xlabel='', ylabel='', add_noise=False, color_on_field=None, xticklabels=['', 1,2,3,4,5], yticklabels=['', 1,2,3,4,5], adjust_vals=False):
    # Create the scatter plot
    plt.figure(figsize=(8, 6)) # Set a good figure size

    if add_noise:
        df = copy.deepcopy(df)
        df[col_x] = df[col_x] + np.random.normal(loc=0, scale=0.05, size=len(df))
        df[col_y] = df[col_y] + np.random.normal(loc=0, scale=0.05, size=len(df))
    if adjust_vals:
        df = copy.deepcopy(df)
        df[col_x] = df[col_x]
        df[col_y] = df[col_y]
        
    g = sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.3, s=45, hue=color_on_field, palette=HUE_COLORS)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    g.set_xticks(range(likert_start,likert_stop))
    g.set_xticklabels(xticklabels)
    g.set_yticks(range(likert_start,likert_stop))
    g.set_yticklabels(yticklabels)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7) # Add a subtle grid for readability

    ax = plt.gca()
    line_coords = np.linspace((likert_start,likert_start), (likert_stop-1 ,likert_stop-1), 100)
    ax.plot(line_coords, line_coords, color='red', linestyle='--', linewidth=2, label='y=x')
    
    handles, labels = ax.get_legend_handles_labels()
    new_handles = handles[:3]
    new_labels = labels[:2] + ['y=x']
    plt.legend(new_handles, new_labels, title='Legend', bbox_to_anchor=(1, 1))
    # plt.legend(title='Legend', bbox_to_anchor=(1, 1))

    plt.show()


def plot_multiple_scatter_with_lines(
    df: pd.DataFrame,
    col_x_list: list[str],
    col_y_list: list[str],
    titles: list[str] = None, # Optional list of titles for each subplot
    xlabel_list = [],
    ylabel_list = [],
    show_diag=True,
    xticks=range(likert_start,likert_stop),
    yticks=range(likert_start,likert_stop),
    xticklbls=['', '1', '2', '3', '4', '5'],
    yticklbls=['', '1', '2', '3', '4', '5'],
    xlims=(0.5, 5.5),
    ylims=(0.5, 5.5),
    color_on=None,
    xfontsize=None
):
    if len(col_x_list) != len(col_y_list):
        raise ValueError("col_x_list and col_y_list must have the same number of elements.")
    if titles is not None and len(titles) != len(col_x_list):
        raise ValueError("If 'titles' is provided, its length must match col_x_list.")

    num_plots = len(col_x_list)
    # Determine grid size: aim for a somewhat square layout
    n_cols = 2 # Or adjust based on num_plots, e.g., int(np.ceil(np.sqrt(num_plots)))
    n_rows = int(np.ceil(num_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False) # Use squeeze=False for consistent 2D array
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    for i in range(num_plots):
        ax = axes[i]
        col_x = col_x_list[i]
        col_y = col_y_list[i]
        title = titles[i] if titles else f"Scatter Plot: {col_x} vs {col_y}"

        # Validate columns exist in the DataFrame for the current pair
        if col_x not in df.columns:
            print(f"Warning: Column '{col_x}' not found. Skipping plot for this pair.")
            ax.set_visible(False) # Hide empty subplot
            continue
        if col_y not in df.columns:
            print(f"Warning: Column '{col_y}' not found. Skipping plot for this pair.")
            ax.set_visible(False) # Hide empty subplot
            continue

        c_l = 'red'
        if color_on:
            c_l = 'purple'
            
        if show_diag:
            # Add the diagonal line (y=x)
            line_coords = np.linspace(1, 5, 100)
            ax.plot(line_coords, line_coords, color=c_l, linestyle='--', linewidth=1, label='y=x', zorder=0)

            handles, labels = ax.get_legend_handles_labels()
            line_handle = [h for i, h in enumerate(handles) if labels[i] == 'y=x']
            line_label = ['y=x']
            ax.legend(line_handle, line_label, loc='upper left', title='Legend', frameon=True)
        else:
            ax.legend(loc='upper left', title='Legend', frameon=True)

        # Create the scatter plot on the current axis
        if color_on:
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, s=50, alpha=0.33, hue=color_on, palette=HUE_COLORS)
        else:
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, s=50, alpha=0.33)

        # Add labels and title to the current subplot
        ax.set_xlabel(xlabel_list[i]) # Use common label or specific column name
        ax.set_ylabel(ylabel_list[i]) # Use common label or specific column name
        ax.set_title(title)

        # Set consistent ticks and labels based on your original request
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklbls, size=xfontsize)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklbls)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set consistent axis limits (important for y=x line)
        ax.set_xlim(xlims[0], xlims[1]) # Add a little padding
        ax.set_ylim(ylims[0], ylims[1])




    # Hide any unused subplots if num_plots doesn't fill the grid perfectly
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() # Adjust layout for all subplots
    plt.show()


def make_scatter_with_regline(per_Q_data, agreelist, yticks, xticks, yticklbls, xticklbls, ylbl, xlbl, title, color_on, x_or='any_dif_from_prev', y_or='understand_exp'):
    tmp_data = copy.deepcopy(per_Q_data)
    tmp_data = tmp_data[~tmp_data[x_or].isna()]
    tmp_data = tmp_data[~tmp_data[y_or].isna()]
    tmp_data['question_number'] = tmp_data['question_number'].apply(lambda x: x.lower().replace('q', 'Case '))
    tmp_data[y_or] = tmp_data[y_or]


    plt.figure(figsize=(10, 7)) # Create a figure to control size

    for iz, z in enumerate(per_Q_data[color_on].unique()):
        cl = HUE_COLORS[z]
        alt_colors = []
        datasubset = tmp_data[tmp_data[color_on] == z]
 
        ax = sns.regplot(
            x=x_or, y=y_or, 
            data=datasubset, 
            line_kws={'color':cl}, 
            scatter_kws={'alpha': 0.2, 'color':cl},
            x_jitter=0.15, 
            y_jitter=0.15,
            # hue=color_on, 
            label=f"{z}"
        )

        #create regression line with text
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            datasubset[x_or], 
            datasubset[y_or])
        equation_text = f'{z} Regression Line\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}'
        plt.text(
            1.05, 0.65 - (iz/5), #location
            equation_text,
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top', 
            label=f"{z} Mean",
            bbox=dict(fc=cl, alpha=0.5)
        )


    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklbls)
    ax.set_ylabel(ylbl)

    ax.set_xticks(xticks)
    ax.set_xlabel(xlbl)
    if xticklbls:
        ax.set_xticklabels(xticklbls)

    ax.set_title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_stripplot(data_used, grouptype, score_col, noise_val=0.05, 
                          fintitle="", xlbl="", ylbl="", ylim_tuple="", 
                          x_tick_labels="", ytick=None, yticklbl=None, xtick=None, xticklbl=None,
                          color_on=None, adjust_score=True):


    temp_df = copy.deepcopy(data_used)
    if adjust_score:
        temp_df[score_col] = add_vertical_noise(temp_df[score_col], noise_val)

    # print(f"color_on: {color_on}, HUE_COLORS:{HUE_COLORS}, ylbl:{ylbl}")
    plt.figure(figsize=(10, 7)) 

    ax_overall = sns.stripplot(data=temp_df, x=None, y=score_col, 
                               hue=color_on, 
                               palette=HUE_COLORS, 
                               alpha=0.4
                            #    label=ylbl
                               )
    ax_overall.set_title(fintitle)
    ax_overall.set_xlabel(xlbl)
    ax_overall.set_ylabel(ylbl)
    ax_overall.set_ylim(ylim_tuple)
    x_pos = 0
    
    
    # Draw a short horizontal line at the for this group\
    if color_on:
        data_subset_ids = temp_df[color_on].unique()
        for v in data_subset_ids:
            subset1 = temp_df[temp_df[color_on] == v]
            mean_val = subset1[score_col].mean()
            ax_overall.plot([x_pos - 0.1, x_pos + 0.1], [mean_val, mean_val],
                color=HUE_COLORS[v], linestyle='dashed', linewidth=2, zorder=3,
                label=f'Mean for {v}') # Label for legend05
            
        handles, labels = ax_overall.get_legend_handles_labels()
        new_handles = [h for i, h in enumerate(handles)][-2:]
        new_labels = ['Mean']
        ax_overall.legend(title='Legend', ncol=1, bbox_to_anchor=(1, 1))
    else:
        mean_val = temp_df[score_col].mean()
        ax_overall.plot([x_pos - 0.1, x_pos + 0.1], [mean_val, mean_val],
                color='red', linestyle='-', linewidth=2, zorder=3,
                label=f'Mean') # Label for legend05
            
        handles, labels = ax_overall.get_legend_handles_labels()
        new_handles = [h for i, h in enumerate(handles)][-1]
        new_labels = ['Mean']
        ax_overall.legend(title='Legend', ncol=1, bbox_to_anchor=(1, 1))
    
    if ytick:
        ax_overall.set_yticks(ytick)
    if yticklbl:
        ax_overall.set_yticklabels(yticklbl)
    if xtick:
        ax_overall.set_xticks(xtick)
    if xticklbl:
        ax_overall.set_xticklabels(xticklbl)



def make_upset_plot(my_data, title, color_on=None):

    from upsetplot import plot, from_indicators

    fig1 = plt.figure(figsize=(5, 5)) # You can set the size for each figure independently
    ax1 = fig1.add_subplot(111)

    if color_on:
        ax_bar_plt = sns.countplot(
            x='post_multi_pref',
            data=my_data,
            ax=ax1,
            # palette='viridis',
            hue=color_on,
            palette=HUE_COLORS,
        )
        sns.move_legend(ax_bar_plt, loc='lower right')
    else:
        ax_bar_plt = sns.countplot(
            x='post_multi_pref',
            data=my_data,
            ax=ax1,
            palette='mako',
            # hue=color_on,
            # palette=HUE_COLORS,
        )
    for p in ax_bar_plt.patches[:4]:
        ax_bar_plt.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=10, color='black')
    ax_bar_plt.set_title('Would you prefer to see multiple types of\nexplanations together for a given patient?', )
    ax_bar_plt.set_ylabel('Count')
    ax_bar_plt.set_xticklabels(['Yes', 'No'])
    ax_bar_plt.set_xlabel('')
    ax_bar_plt.set_xticks([0,1])


    if color_on:
        for x in my_data[color_on].unique():
            subset_data = my_data[my_data[color_on] == x]
            temp_data = subset_data[subset_data['post_multi_pref'] == 1]
            df_bool = temp_data[['Attribution explanations', 'Counterfactual explanations', 'Rule-based explanations']].astype(bool)
            upset_data = from_indicators(df_bool)

            x_color = HUE_COLORS[x]



            fig2 = plt.figure(figsize=(5, 5))
            plot(upset_data,
                fig=fig2,
                #  subplot_spec=gs[0, 1],
                orientation='horizontal',
                show_counts=True,
                sort_by='cardinality',
                facecolor=x_color,
                element_size=30,
                intersection_plot_elements=5,
                subset_size="count",
                )

            plt.suptitle(f"{x} - {title}", fontsize=14, y=1.05)
            plt.ylabel("Intersection Size", fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.show()

    else:
        temp_data = my_data[my_data['post_multi_pref'] == 1]
        df_bool = temp_data[['Attribution explanations', 'Counterfactual explanations', 'Rule-based explanations']].astype(bool)
        upset_data = from_indicators(df_bool)

        fig2 = plt.figure(figsize=(5, 5))
        plot(upset_data,
            fig=fig2,
            #  subplot_spec=gs[0, 1],
            orientation='horizontal',
            show_counts=True,
            sort_by='cardinality',
            facecolor='blue',
            element_size=30,
            intersection_plot_elements=5,
            subset_size="count",
            )

        plt.suptitle(title, fontsize=14, y=1.05)
        plt.ylabel("Intersection Size", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()


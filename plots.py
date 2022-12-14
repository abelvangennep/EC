import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# def line_plot(mean_csv_EA1, max_csv_EA1):
#     mean_df = pd.read_csv(mean_csv_EA1)
#     max_df = pd.read_csv(max_csv_EA1)
#     mean_df['mean'] = mean_df.mean(axis=1)
#     mean_df['mean_std'] = mean_df.std(axis=1)
#     max_df['max'] = max_df.mean(axis=1)
#     max_df['max_std'] = max_df.std(axis=1)
#     df = mean_df[['mean', 'mean_std']]
#     df['max'] = max_df['max'].values
#     df['max_std'] = max_df['max_std'].values
#     # df.index.name = "Generation"
#
#     seq = range(1,21)
#     df['Generation'] = seq
#     df_melted = df.melt('Generation', ['mean','max'], value_name='value')
#     fig, ax = plt.subplots(figsize=(9, 5))
#
#     # # ax = sns.relplot(x='Generation', y='mean', data=df, ci='kind')
#     # # ax1 = sns.lineplot(x='Generation', y='max', data=df, ci='sd')
#     # plt.plot(x='Generation', y='mean',data=df)
#     # plt.plot(x='Generation', y='max', data=df)
#     # # plt.errorbar(x='Generation', y='mean', error= 'mean_std', data=df, color='tab:blue', ecolor='tab:blue')
#     # plt.legend(loc='upper left', labels=['Mean', 'Max'])
#     # sns.relplot(data=df_melted, x='Generation', y ='value', hue='variable', style='variable', kind ='line')
#     # plt.ylabel('Fitness')
#     # plt.xlabel('Generation')
#     # sns.despine(ax=ax)
#     # sns.despine(ax=ax1)

#       return print(df)

# line_plot('mean_fitness_10runs_enemy4.csv', 'max_fitness_10runs_enemy4.csv')

def lineplot(max_fit_csv_EA1, mean_fit_csv_EA1, mean_fit_csv_EA2, max_fit_csv_EA2):
    plot_mean_fit = np.loadtxt(mean_fit_csv_EA1, delimiter = ',', skiprows=1)[:,1:]
    plot_max_fit = np.loadtxt(max_fit_csv_EA1, delimiter = ',', skiprows=1)[:,1:]
    plot_mean_fit_2 = np.loadtxt(mean_fit_csv_EA2, delimiter = ',', skiprows=1)[:,1:]
    plot_max_fit_2 = np.loadtxt(max_fit_csv_EA2, delimiter = ',', skiprows=1)[:,1:]
    number_generations = len(plot_max_fit)
    x = [i for i in range(1,number_generations+1)]

    y1_EA1 = np.average(plot_mean_fit[:], axis=1)
    y2_EA1 = np.average(plot_max_fit[:], axis=1)
    y1_EA2 = np.average(plot_mean_fit_2[:], axis=1)
    y2_EA2 = np.average(plot_max_fit_2[:], axis=1)


    y1std_EA1 = np.std(plot_mean_fit, axis = 1)
    y2std_EA1 = np.std(plot_max_fit, axis = 1)
    y1std_EA2 = np.std(plot_mean_fit_2, axis = 1)
    y2std_EA2 = np.std(plot_max_fit_2, axis = 1)

    fig = plt.subplots(figsize=(15, 9))

    plt.plot(x,y1_EA1)
    plt.plot(x,y2_EA1)
    plt.plot(x,y1_EA2)
    plt.plot(x,y2_EA2)
    # plt.errorbar(x,y1,y1std)
    # plt.errorbar(x,y2,y2std)
    plt.fill_between(x, y1_EA1 - y1std_EA1, y1_EA1 + y1std_EA1, alpha=0.5)
    plt.fill_between(x, y2_EA1 - y2std_EA1, y2_EA1 + y2std_EA1, alpha=0.5)
    plt.fill_between(x, y1_EA2 - y1std_EA2, y1_EA2 + y1std_EA2, alpha=0.5)
    plt.fill_between(x, y2_EA2 - y2std_EA2, y2_EA2 + y2std_EA2, alpha=0.5)

    plt.legend(loc='upper left', labels=['Max EA1', 'Mean EA1'])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')

    return plt.show()



def boxplot(boxplot_EA1, boxplot_EA2):
    EA1 = np.loadtxt(boxplot_EA1, delimiter = ',', skiprows = 1)[:,1:]
    EA2 = np.loadtxt(boxplot_EA2, delimiter = ',', skiprows = 1)[:,1:]
    EA1 = [EA1[i, 0] for i in range(len(EA1))]
    EA2 = [EA2[i, 0] for i in range(len(EA2))]
    data = {'EA 1': EA1,
            'EA 2': EA2}
    #data = [EA1, EA2]
    df = pd.DataFrame(data)
    #df = pd.DataFrame(data, columns=['EA1', 'EA2'])


    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(df)
    plt.xlabel('EA')
    plt.ylabel('Individual Gain')
    plt.xticks([1, 2], ['EA 1', 'EA 2'])
    return plt.show()

if __name__ == '__main__':
   boxplot('EA_1/boxplot_NEAT110runs_enemy_group2.csv', 'EA_2/boxplot_EA210runs_enemy2.csv')
   #lineplot('EA_1/mean_fitness_EA110runs_enemy_group2.csv', 'EA_1/max_fitness_EA110runs_enemy_group2.csv', 'EA_2/mean_fitness_EA210runs_enemy2.csv', 'EA_2/max_fitness_EA210runs_enemy2.csv')

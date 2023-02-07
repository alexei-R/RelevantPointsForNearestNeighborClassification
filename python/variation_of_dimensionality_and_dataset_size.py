import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("../measurements/variation_of_dimensionality_and_dataset_size.csv", encoding='utf_16_le', delimiter = ';')

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for d in range(3, 12, 2):
        split = data[data['dimensionality'] == d]
        plt.scatter(split['dataset_size'], split['execution_time_eppstein_ms'] / 1000.0, label='Eppstein, $d = ' + str(d) + '$')

    plt.xlabel('$n$')
    plt.ylabel('$t_1$ in s')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('plot_eppstein_execution_time_vs_dataset_size.pdf', bbox_inches = 'tight')
    plt.show()

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for d in range(3, 12, 2):
        split = data[data['dimensionality'] == d]
        plt.scatter(split['dataset_size'], split['execution_time_flores_ms'] / 1000.0, label='Flores-Velazco, $d = ' + str(d) + '$')

    plt.xlabel('$n$')
    plt.ylabel('$t_2$ in s')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('plot_flores_execution_time_vs_dataset_size.pdf', bbox_inches = 'tight')
    plt.show()

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for d in range(3, 12, 2):
        split = data[data['dimensionality'] == d]
        plt.scatter(split['dataset_size'], (split['execution_time_eppstein_ms'] - split['execution_time_flores_ms'])/ 1000.0, label='Differenz, $d = ' + str(d) + '$')

    plt.xlabel('$n$')
    plt.ylabel('$\Delta t = t_1 - t_2$, in s')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('plot_difference_in_execution_time_vs_dataset_size.pdf', bbox_inches = 'tight')
    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("../measurements/variation_of_dimensionality_and_label_consolidation_level.csv", encoding='utf_16_le', delimiter = ';')

legend_labels = ('Ursprüngliche Klassen', 
    'Konsolidierte Klassen mit 0 - 6 → 1, 7 - 10 → 2',
    'Konsolidierte Klassen mit 0 - 7 → 1, 8 - 10 → 2')

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for label_consolidation_level in range(3):
        split = data[data['label_consolidation_level'] == label_consolidation_level]
        plt.scatter(split['dimensionality'], split['boundary_points_flores'] / split['dataset_size'], label=legend_labels[label_consolidation_level])

    ax.xaxis.set_ticks((3, 5, 7, 9, 11))
    plt.xlabel('$d$')
    plt.ylabel('$k / n$')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plot_border_points_fraction_variation_with_level_consolidation_and_dimensionality.pdf', bbox_inches = 'tight')
    plt.show()

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for d in range(3, 12, 2):
        split = data[data['dimensionality'] == d]
        plt.scatter(split['boundary_points_flores'], split['execution_time_eppstein_ms'] / 1000.0, label='Eppstein, $d = ' + str(d) + '$')

    plt.xlabel('$k$')
    plt.ylabel('$t$ in s')
    ax.set_yscale('log')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('plot_eppstein_execution_time_vs_border_point_count.pdf', bbox_inches = 'tight')
    plt.show()

with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    for d in range(3, 12, 2):
        split = data[data['dimensionality'] == d]
        plt.scatter(split['boundary_points_flores'], split['execution_time_flores_ms'] / 1000.0, label='Flores-Velazco, $d = ' + str(d) + '$')

    plt.xlabel('$k$')
    plt.ylabel('$t$ in s')
    ax.set_yscale('log')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('plot_flores_execution_time_vs_border_point_count.pdf', bbox_inches = 'tight')
    plt.show()
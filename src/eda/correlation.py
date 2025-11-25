
import seaborn as sns; import matplotlib.pyplot as plt
def plot_corr(df):
    sns.heatmap(df.corr()); plt.show()

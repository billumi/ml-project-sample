
import seaborn as sns; import matplotlib.pyplot as plt
def plot_hist(df,col): sns.histplot(df[col], kde=True); plt.show()

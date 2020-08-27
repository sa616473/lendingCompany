import seaborn as sns
import matplotlib.pyplot as plt

def visualization_plot(feature_x, data, feature_y=None,  
                       hue=None, kde_line=False, size=(14,7),
                       bins=100, plot_type='', order=None,palette=None):
    '''
    This function plots the specified plot_type if the 
    plot type does not exist it will raise a error.
    plot_type should in the fromat of a string
    '''
    plt.figure(figsize=size)
    
    
    if feature_y == None:
        plt.title("{}_plot of {}".format(plot_type,feature_x))
    else:
        plt.title("{} vs {}".format(feature_x, feature_y))

    if plot_type == 'dist' or plot_type == 'distribution':  
        sns.distplot(data[feature_x],bins=bins, kde=kde_line)
    
    elif plot_type == 'count':
        sns.countplot(x=feature_x, data=data, hue=hue, order=order,palette=palette)
    
    elif plot_type == 'violin' and feature_y != None:
        sns.violinplot(x=feature_x, y=feature_y, data=data)
    
    elif plot_type == 'bar':
        sns.barplot(x=feature_x, y=feature_y, data=data)
    
    elif plot_type == 'box' and feature_y !=None:
        sns.boxplot(x=feature_x, y=feature_y, data=data)
    
    elif plot_type == 'scatter' and feature_y != None:
        sns.scatterplot(x=feature_x, y=feature_y, data=data, alpha=0.5)
    
    else:
        raise ValueError('Invalid plot type\n or\n the type of plot does not exist in the method\n')
        return -1
    if feature_y == None:
        plt.savefig('../reports/figures/graphs/png/{}_{}.png'.format(feature_x, plot_type))
        plt.savefig('../reports/figures/graphs/pdf/{}_{}.pdf'.format(feature_x, plot_type))
    elif feature_y != None:
        plt.savefig('../reports/figures/graphs/png/{}_{}_{}.png'.format(feature_x,feature_y, plot_type))
        plt.savefig('../reports/figures/graphs/pdf/{}_{}_{}.pdf'.format(feature_x,feature_y, plot_type))
        
def corr_plot(data, annot=True, cmap='viridis', size=(12,7), ylim=(13,0)):
    '''
    plots the heat map of the correalation
    '''
    plt.figure(figsize=size)
    plt.title('Correlation plot')
    sns.heatmap(data, annot=annot, cmap=cmap)
    plt.ylim(ylim[0], ylim[1])
    plt.savefig('../reports/figures/graphs/png/corr.png')
    plt.savefig('../reports/figures/graphs/pdf/corr.pdf')

def confusion_plot(cm, size=(12,7), title=''):
    print(cm)
    plt.figure(figsize=size)
    plt.title('{}_confusion_matrix'.format(title))
    sns.heatmap(cm, annot=True)
    
    plt.savefig('../reports/figures/graphs/pdf/{}_confusion_matrix.pdf'.format(title))
    plt.savefig('../reports/figures/graphs/png/{}_confusion_matrix.png'.format(title))

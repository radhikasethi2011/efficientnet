import torch
import matplotlib
import matplotlib.pyplot as plt
from warnings import filterwarnings
import seaborn as sns


colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']


matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"/content/efficientnet/pytorch/EfficientnetB0_mydata_156mb/outputs/model.pth")
def save_plots(epoch, train_acc, val_acc, train_loss, val_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    epochs = [i for i in range(epoch)]
    # accuracy plots
    fig, ax = plt.subplots(1,2,figsize=(14,7))


    fig.text(s='Epochs vs. Training and Validation Accuracy/Loss',size=18,fontweight='bold',
                fontname='monospace',color=colors_dark[1],y=1,x=0.28,alpha=0.8)

    sns.despine()
    ax[0].plot(epochs, train_acc, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
              label = 'Training Accuracy')
    ax[0].plot(epochs, val_acc, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
              label = 'Validation Accuracy')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    sns.despine()
    ax[1].plot(epochs, train_loss, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
              label ='Training Loss')
    ax[1].plot(epochs, val_loss, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
              label = 'Validation Loss')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Training & Validation Loss')

    fig.savefig(f"/content/efficientnet/pytorch/EfficientnetB0_mydata_156mb/outputs/loss.png")

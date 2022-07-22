import matplotlib.pyplot as plt
import os
def visualization(train_loss, valid_loss, train_acc, valid_acc, ROOT):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    epoch_train = len(train_loss)
    epoch_valid = len(valid_loss)
    interval = epoch_train//epoch_valid
    plt.subplot(1,2,1)
    plt.plot(range(epoch_train), train_loss, color='blue', label='train-loss') 
    plt.plot([int(x)*interval for x in range(epoch_valid)], valid_loss, color='yellow', label='valid-loss')
    plt.ylabel('当前损失值')
    plt.xlabel('迭代次数/次')
    plt.legend()                 #添加曲线注释到图中
    plt.subplot(1,2,2)
    plt.plot(range(epoch_train), train_acc, color='red', label='train-acc')
    plt.plot([int(x)*interval for x in range(epoch_valid)], valid_acc, color='green', label='valid-acc')
    plt.legend()
    plt.savefig(os.path.join(ROOT, 'result_visual.jpg')) # 保存曲线图片

visualization([20,17,15,13,9,5,2,2,1], [25, 15, 4], [0.1,0.2,0.22,0.3,0.31,0.4,0.45,0.45,0.6], [0.08, 0.35, 0.52], 'checkpoint')
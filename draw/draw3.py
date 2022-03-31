import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # plt.figure(figsize=(13, 4))
    seed = 10
    alpha = 0.05
    plt.figure()
    ax = plt.subplot(111)
    # 构造x轴刻度标签、数据
    labels = ['FedAvg', 'RFA', 'FoolsGold', 'RFL-DA(ours)']
    # 10 0.05
    acc = [79.6, 78.1, 72.7, 85.6]
    asr = [90.6, 82.6, 0.5, 0.5]

    # 30 0.5
    # acc = [92.5, 91.9, 95.1, 94.9]
    # asr = [29.9, 35.5, 1.56, 3.4]

    # 60 1
    # acc = [93.5, 94.7, 84.9, 94.4]
    # asr = [18.7, 6.7, 2.1, 5.4]

    # 20 10
    # acc = [95.4, 95.8, 95.9, 95.8]
    # asr = [8.1, 4.0, 1.2, 1.1]

    # third = [21, 31, 37, 21, 28]
    # fourth = [26, 31, 35, 27, 21]

    # 两组数据
    # plt.subplot(131)
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plot1 = ax.bar(x - width / 2, acc, width, label='Accuracy')
    plot2 = ax.bar(x + width / 2, asr, width, label='Attack Success Rate')

    for value in plot1:
        height = value.get_height()
        ax.text(value.get_x() + value.get_width() / 2.,
                1.002 * height, '%.1f' % height, ha='center', va='bottom')
    for value in plot2:
        height = value.get_height()
        ax.text(value.get_x() + value.get_width() / 2.,
                1.002 * height, '%.1f' % height, ha='center', va='bottom')

    ax.set_ylabel("Rate(%)")
    # ax.ylabel('Rate(%)')
    # plt.title('2 datasets')
    # x轴刻度标签位置不进行计算
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=False, ncol=5)
    plt.savefig(f"../imgs/dp_seed{seed}_alpha{alpha}.svg")
    # plt.legend(frameon=False)
    # 三组数据
    # plt.subplot(132)
    # x = np.arange(len(labels))  # x轴刻度标签位置
    # width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width，x， x + width即每组数据在x轴上的位置
    # plt.bar(x - width, first, width, label='1')
    # plt.bar(x, second, width, label='2')
    # plt.bar(x + width, third, width, label='3')
    # plt.ylabel('Scores')
    # plt.title('3 datasets')
    # x轴刻度标签位置不进行计算
    # plt.xticks(x, labels=labels)
    # plt.legend()
    # # 四组数据
    # plt.subplot(133)
    # x = np.arange(len(labels))  # x轴刻度标签位置
    # width = 0.2  # 柱子的宽度
    # # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # plt.bar(x - 1.5 * width, first, width, label='1')
    # plt.bar(x - 0.5 * width, second, width, label='2')
    # plt.bar(x + 0.5 * width, third, width, label='3')
    # plt.bar(x + 1.5 * width, fourth, width, label='4')
    # plt.ylabel('Scores')
    # plt.title('4 datasets')
    # # x轴刻度标签位置不进行计算
    # plt.xticks(x, labels=labels)
    # plt.legend()

    # plt.show()

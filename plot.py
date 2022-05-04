from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = Path(os.path.realpath(__file__)).parent

plt.style.use('seaborn-white')

# # *********************************** data **********************************************
# acc-mnist
# x = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]).astype('str')
#
# y0 = [88.43999481201172, 96.69999694824219, 97.14999389648438, 97.23999786376953, 97.18999481201172, 96.97999572753906,
#       97.25999450683594, 97.43999481201172, 97.31999969482422, 97.48999786376953]
#
# y1 = [96.66999816894531, 96.88999938964844, 98.00999450683594, 96.97999572753906, 97.06999969482422, 97.69999694824219,
#       97.0999984741211, 97.32999420166016, 97.66999816894531, 97.61000061035156]

# acc-cifar
x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
     61,
     63, 65, 67, 69, 71, 73, 75, 77, 79]
y0 = [28.939998626708984, 37.439998626708984, 40.209999084472656, 43.939998626708984, 45.29999923706055,
      47.04999923706055,
      48.119998931884766, 50.18000030517578, 52.03999710083008, 52.959999084472656, 53.38999938964844,
      54.64999771118164,
      55.03999710083008, 55.91999816894531, 55.84000015258789, 56.869998931884766, 56.189998626708984,
      55.88999938964844,
      57.689998626708984, 57.119998931884766, 58.44999694824219, 58.34000015258789, 58.369998931884766,
      58.84000015258789,
      59.21999740600586, 59.40999984741211, 60.40999984741211, 59.56999969482422, 59.59000015258789, 60.279998779296875,
      60.81999969482422, 60.25, 59.959999084472656, 60.53999710083008, 61.07999801635742, 61.459999084472656,
      61.459999084472656, 60.75, 61.54999923706055, 61.90999984741211]
y1 = [46.88999938964844, 53.12999725341797, 56.0099983215332, 56.75, 58.57999801635742, 60.119998931884766,
      59.40999984741211, 61.07999801635742, 60.09000015258789, 61.47999954223633, 62.15999984741211, 59.81999969482422,
      62.22999954223633, 63.55999755859375, 62.439998626708984, 63.2599983215332, 63.19999694824219, 62.68000030517578,
      63.15999984741211, 63.72999954223633, 63.69999694824219, 62.18000030517578, 64.91999816894531, 62.939998626708984,
      63.78999710083008, 64.11000061035156, 63.88999938964844, 65.40999603271484, 63.849998474121094, 64.68000030517578,
      64.97999572753906, 62.75, 64.73999786376953, 64.5199966430664, 64.0, 65.47000122070312, 64.63999938964844,
      65.68999481201172, 65.58000183105469, 65.04000091552734]

# *********************************** draw **********************************************
font_family = 'Times New Roman'
font_size = 26
font_dict = {'family': font_family, 'size': font_size}
plt.figure(figsize=(12, 8))

plt.tick_params(labelright=True)

plt.plot(x, y0, 'rs-', label='DemoFL')
plt.plot(x, y1, 'bs-', label='CNN')

plt.xlabel('Epoch', fontdict=font_dict)
plt.ylabel('Accuracy (%)', fontdict=font_dict)

plt.xticks(fontproperties=font_family, size=font_size - 2)
plt.yticks(fontproperties=font_family, size=font_size - 2)

plt.legend(prop={'family': font_family, 'size': font_size})
# plt.show()

plt.savefig(BASE_PATH / r'figure/acc-cifar.pdf', bbox_inches='tight')

# print(plt.style.available)

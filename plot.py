from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = Path(os.path.realpath(__file__)).parent

plt.style.use('seaborn-white')

# *********************************** data **********************************************
x = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]).astype('str')

y0 = [88.43999481201172, 96.69999694824219, 97.14999389648438, 97.23999786376953, 97.18999481201172, 96.97999572753906,
      97.25999450683594, 97.43999481201172, 97.31999969482422, 97.48999786376953]

y1 = [96.66999816894531, 96.88999938964844, 98.00999450683594, 96.97999572753906, 97.06999969482422, 97.69999694824219,
      97.0999984741211, 97.32999420166016, 97.66999816894531, 97.61000061035156]

# *********************************** draw **********************************************
font_family = 'Times New Roman'
font_size = 26
font_dict = {'family': font_family, 'size': font_size}
plt.figure(figsize=(12, 8))

plt.tick_params(labelright=True)

plt.plot(x, y0, 'rs-', label='DemoFL')
plt.plot(x, y1, 'bs-', label='Pure CNN')

plt.xlabel('Epoch', fontdict=font_dict)
plt.ylabel('Accuracy (%)', fontdict=font_dict)

plt.xticks(fontproperties=font_family, size=font_size - 2)
plt.yticks(fontproperties=font_family, size=font_size - 2)

plt.legend(prop={'family': font_family, 'size': font_size})
plt.show()

# plt.savefig(BASE_PATH / r'figure/acc.pdf', bbox_inches='tight')

print(plt.style.available)

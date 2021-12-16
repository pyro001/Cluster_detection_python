import numpy as np
import matplotlib.pyplot as plt

data = np.random.randint(0, 10, 30)

num_foreground_pixels = np.random.randint(100, 400, 16)
num_circles = np.random.randint(0, 8, 16)
num_rods = np.random.randint(0, 8, 16)
num_triangles = np.random.randint(0, 8, 16)

test_list = [[1, 2, 3], [4, 5, 6]]
print(test_list)
test_list = np.array(test_list)
print(test_list)
print(test_list.size)

# print(data)

# plt.hist(data)
# plt.show()

# plt.boxplot([num_foreground_pixels, num_circles])
# plt.show()

# plt.violinplot(data)
# plt.show()


# axs = plt.subplots(3, 1)

# axs[0].boxplot(data)
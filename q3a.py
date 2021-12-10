from pandas import DataFrame as df, Series
import re
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import amin, amax, ones, zeros

content = open('score.txt', "r")
lines = content.readlines()
content.close()

# Data cleansing

# remove /n at the end of each line and split the data by spaces
for index, line in enumerate(lines):
    lines[index] = line.strip().split()

for line in lines[1:]:
    line[0] = float(line[0])
    line[1] = int(line[1])
    line[2] = int(line[2])

# Data representation

df_result = df(columns=lines[0], data=lines[1:])

print(df_result.head())

n = df_result.shape[0]

# Now z_{ij} = b_0 + u_i + n_j + e_{ij}

_1 = ones((n, 1)) # 450 school-subject combinations bruh
W_u = zeros((n, 30)) # create 450*30 empty matrice first
W_n = zeros((n, 15))  # create 450*15 empty matrice first

# u follows N_30(zeros(30, 1), var_u * I)       # not know
# n follows N_15(zeros(15, 1), var_n * I)       # not know
# e follows N_3000(zeros(3000, 1), var_e * I)   # not know

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Create the counts
schools = range(1, 31)
subjects = range(1, 16)
counts = zeros((15, 30))

# fill in the counts
for index, line in enumerate(lines):
    if index > 0:
        u, v = lines[index][1], lines[index][2]
        counts[v-1, u-1] += 1
        # W_u = random effect by school             # to be filled up
        W_u[index-1, u-1] = 1
        # W_n = random effect by subject            # to be filled up
        W_n[index-1, v-1] = 1

counts_df = df(data=counts, columns=schools, index=subjects)

# Create heatmap
counts_map = counts_df.to_numpy().astype(float)

figa, axea = plt.subplots(figsize=(8.5, 4.5))
axea = sns.heatmap(counts_map, annot=True, linewidths=.5, square=True,
                   xticklabels=schools, yticklabels=subjects,
                   vmin=amin(counts_map), vmax=amax(counts_map),
                   cmap='Reds', cbar=False)
axea.set_xlabel("School")
axea.set_ylabel("Subject")
axea.set_title("School-subject counts")
figa.tight_layout()

plt.yticks(rotation=0)

plt.show()
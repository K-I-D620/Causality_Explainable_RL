import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# To change to curr dir of python script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

file_path_CFRL_iter = "./0_CFRL_iter_monitor_stack.csv"

df_CFRL_iter = pd.read_csv(file_path_CFRL_iter, skiprows=2, names=['Reward', 'Len_of_eps', 'Time_elapsed', 'Frac_success'])
# Print first 5 rows of df
# print(df_noIntervene.Frac_success.head)
np_timesteps_CFRL_iter = np.arange(1667, 6978062, 1667)
# print(len(np_timesteps_noInter))
#print(np_timesteps_noInter[:5])
ind_timesteps_CFRL_iter = len(np_timesteps_CFRL_iter) # len = 4186 episodes

# Get only that portion of timesteps up till a little pass 7000 000
df_CFRL_iter = df_CFRL_iter[:ind_timesteps_CFRL_iter]
df_CFRL_iter['Timesteps'] = np_timesteps_CFRL_iter
# print(df_noIntervene.head)

# Get mean of fractional success for every 100 episodes or 166 700 time steps, because graph have rapid changes.
np_frac_success_CFRL_iter = np.array(df_CFRL_iter.Frac_success)
lower_index = 0
upper_index = 200
np_mean_100eps_FS_CFRL_iter = np.zeros(21)
for i in range(21):

    if upper_index < 4186:
        np_mean_100eps_FS_CFRL_iter[i] = np.mean(np_frac_success_CFRL_iter[lower_index:upper_index])
    else:
        np_mean_100eps_FS_CFRL_iter[i] = np.mean(np_frac_success_CFRL_iter[lower_index:])

    lower_index += 200
    upper_index += 200

np_timesteps_mean_FS = np.arange(333400, 7334800, 333400)
# print(np_mean_100eps_FS_Inter[:5])
# print(len(np_mean_100eps_FS_Inter))
# print(len(np_timesteps_mean_FS))

# Plot graph
plt.plot(np_timesteps_mean_FS, np_mean_100eps_FS_CFRL_iter)
plt.title('Stacking (train)', fontsize=10)
plt.xlabel('Time steps')
plt.ylabel('Mean fractional success (200 eps)')
plt.legend(['CausalCF (iter)'], bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True)
plt.tight_layout()
plt.savefig('./Stacking_training_iter_2.png')
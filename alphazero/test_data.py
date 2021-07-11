import numpy as np
from database import DataBase

db = DataBase(max_len=10000)
ind1, ind2 = 82, 87

final_sm = np.zeros((10, 10))
final_pol = np.zeros((10, 10))

for num in range(ind1, ind2):
    db.load_data("/storage/replay_buffer", num)
    sm = np.zeros((10, 10))
    pol = np.zeros((10, 10))
    for i, st in enumerate(db.states):
        if i % 8 == 0:
            sm += st
            pol += db.policy_labels[i].reshape((10, 10))

    # sm /= 10000.0
    # pol /= 10000.0
    sm /= (10000.0 / 8.0)
    pol /= (10000.0 / 8.0)

    final_sm += sm
    final_pol += pol

final_sm /= ind2 - ind1
final_pol /= ind2 - ind1

print(np.around(final_sm, 2))
print(np.around(final_pol, 2))

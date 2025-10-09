# Lab 2: Monte Carlo method implementation for Mountain Car scenario

Discretization params:
`position_bins=50`
`velocity_bins=25`
`n_value_table_iter=50`(for Policy iteration algorithm)

### Monte Carlo method results:

Reward function: `potential_factor * pos_normalized + kinetic_factor * vel_normalized`

`potential_factor=0.55`
`kinetic_factor=0.45`
`pos_normalized є [0, 1]`
`vel_normalized є [0, 1]`

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 5000             | -354             |
| 10000            | -264             |
| 15000            | -188             |
| 20000            | -306             |


Reward function: `-1 + np.abs(next_pos - pos)`

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 5000             | -1000            |
| 10000            | -1000            |

Reward function: `(position + velocity ** 2) - 1`

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 5000             | -1000            |
| 10000            | -1000            |

Reward function: `(position + 200 * velocity ** 2) - 1`

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 5000             | -1000            |
| 10000            | -205             |
| 15000            | -1000            |
| 20000            | -160             |


### Value iteration algorithm previous results(look Lab 1)

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 10               | -157             |
| 20               | -112             |
| 50               | -181             |
| 75               | -143             |
| 100              | -118             |
| 200              | -111             |
| 300              | -110             |
| 400              | -111             |
| 500              | -113             |


### Policy iteration algorithm previous results(look Lab 1)

| **№ Iterations** | **Total reward** |
|------------------|------------------|
| 2                | -121             |
| 5                | -147             |
| 10               | -140             |
| 15               | -113             |
| 20               | -114             |
| 25               | -113             |
| 30               | -114             |

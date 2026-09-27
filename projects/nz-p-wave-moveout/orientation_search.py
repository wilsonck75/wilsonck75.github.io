import numpy as np
import matplotlib
matplotlib.use('Agg')   # no gui window needed
import matplotlib.pyplot as plt

# load the data
data = np.load('tohoku_nz_array.npz', allow_pickle=True)

Z         = data['Z']
R         = data['R']
T         = data['T']
distances = data['distances']
times     = data['times_rel']
stations  = data['stations']
sr        = float(data['sampling_rate'])
baz_cat   = data['baz']

# sort by distance
order     = np.argsort(distances)
Z         = Z[order]
R         = R[order]
T         = T[order]
distances = distances[order]
stations  = stations[order]
baz_cat   = baz_cat[order]

num_stations = len(stations)

# search for P peak on Z in 0 to 10 seconds after predicted onset
p_search_start = 0.0
p_search_end   = 10.0

# length of analysis window after the P pick
win_len = 3.0   # seconds

# go through every azimuth from 0 to 359
azimuths_to_try = np.arange(0, 360, 1)

# store results
pca_baz        = np.zeros(num_stations)
gridsearch_baz = np.zeros(num_stations)

for i in range(num_stations):

    # ---- find P pick on Z ----
    i_search_start = int(round((p_search_start - times[0]) * sr))
    i_search_end   = int(round((p_search_end   - times[0]) * sr))
    p_peak_offset  = np.argmax(np.abs(Z[i, i_search_start:i_search_end]))
    p_idx          = i_search_start + p_peak_offset

    # window: from the P pick forward
    win_start = p_idx
    win_end   = min(Z.shape[1], p_idx + int(round(win_len * sr)))

    # ---- step 1: back-rotate R/T to N/E using the catalog back-azimuth ----
    # download_data.py used rotate_ne_rt which does:
    #   R = -N*cos(baz) - E*sin(baz)
    #   T =  N*sin(baz) - E*cos(baz)
    # so the inverse is:
    baz_rad = np.radians(baz_cat[i])
    N_comp  = -R[i] * np.cos(baz_rad) + T[i] * np.sin(baz_rad)
    E_comp  = -R[i] * np.sin(baz_rad) - T[i] * np.cos(baz_rad)

    # grab just the p window samples
    z_window = Z[i, win_start:win_end]
    n_window = N_comp[win_start:win_end]
    e_window = E_comp[win_start:win_end]

    # ---- step 2: PCA ----
    # stack the three components into a matrix and compute covariance
    data_matrix = np.column_stack([z_window, n_window, e_window])
    data_matrix = data_matrix - data_matrix.mean(axis=0)
    cov_matrix  = data_matrix.T @ data_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # biggest eigenvalue = dominant direction of motion
    best_idx = np.argmax(eigenvalues)
    pc1      = eigenvectors[:, best_idx]

    # the sign of Z at the very first sample tells us the true first motion polarity
    # (argmax can land on a later negative cycle)
    first_z_sign = np.sign(Z[i, win_start])
    if first_z_sign == 0:
        first_z_sign = 1.0
    if np.sign(pc1[0]) != first_z_sign:
        pc1 = -pc1

    # back-azimuth is the direction the horizontal part of pc1 points
    # arctan2(east, north) gives angle from north
    pca_baz[i] = np.degrees(np.arctan2(pc1[2], pc1[1])) % 360

    # ---- step 3: grid search ----
    # try every azimuth, rotate N/E to radial, measure energy on radial
    # the correct back-azimuth puts most energy on radial and least on transverse
    # note: az and az+180 give same energy so we use first-sample sign to pick
    best_energy = -1.0
    best_az     = 0.0

    for az in azimuths_to_try:
        az_rad = np.radians(az)

        # rotate to radial for this trial azimuth
        # same formula as rotate_ne_rt
        r_trial = -N_comp * np.cos(az_rad) - E_comp * np.sin(az_rad)

        # energy in the p window on radial
        r_energy = np.sum(r_trial[win_start:win_end] ** 2)

        if r_energy > best_energy:
            best_energy = r_energy
            best_az     = az

    # break the 180 deg ambiguity: the correct azimuth has R first motion
    # with the same sign as Z first motion (both pointing away from source)
    az_rad_best = np.radians(best_az)
    r_best = -N_comp * np.cos(az_rad_best) - E_comp * np.sin(az_rad_best)
    r_first_sign = np.sign(r_best[win_start])
    if r_first_sign != 0 and r_first_sign != first_z_sign:
        best_az = (best_az + 180) % 360

    gridsearch_baz[i] = best_az

# ---- print results ----
def wrap180(angle):
    # wrap angle to -180 to +180
    return ((angle + 180) % 360) - 180

print(f"{'Station':<12}  {'Catalog':>8}  {'PCA':>8}  {'GridSearch':>11}  {'dPCA':>7}  {'dGrid':>7}")
print("-" * 65)
for i in range(num_stations):
    d_pca  = wrap180(pca_baz[i]        - baz_cat[i])
    d_grid = wrap180(gridsearch_baz[i] - baz_cat[i])
    print(f"{stations[i]:<12}  {baz_cat[i]:>8.1f}  {pca_baz[i]:>8.1f}  "
          f"{gridsearch_baz[i]:>11.1f}  {d_pca:>+7.1f}  {d_grid:>+7.1f}")

# ---- plot ----
delta_pca  = np.array([wrap180(pca_baz[i]        - baz_cat[i]) for i in range(num_stations)])
delta_grid = np.array([wrap180(gridsearch_baz[i] - baz_cat[i]) for i in range(num_stations)])

x = np.arange(num_stations)
sta_labels = [s.split('.')[1] for s in stations]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].bar(x, delta_pca, color='steelblue', label='PCA')
axes[0].axhline(0,   color='black', linewidth=0.8)
axes[0].axhline( 20, color='gray',  linewidth=0.8, linestyle='--')
axes[0].axhline(-20, color='gray',  linewidth=0.8, linestyle='--')
axes[0].set_ylabel('PCA baz - catalog (°)')
axes[0].set_title('Back-azimuth difference from catalog')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x, delta_grid, color='#e67e22', label='Grid search')
axes[1].axhline(0,   color='black', linewidth=0.8)
axes[1].axhline( 20, color='gray',  linewidth=0.8, linestyle='--')
axes[1].axhline(-20, color='gray',  linewidth=0.8, linestyle='--')
axes[1].set_ylabel('Grid baz - catalog (°)')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

axes[1].set_xticks(x)
axes[1].set_xticklabels(sta_labels, rotation=45, ha='right', fontsize=8)

fig.suptitle('PCA vs Grid Search back-azimuth  —  2011 Tohoku / GeoNet NZ', fontsize=12)
plt.tight_layout()
plt.savefig('orientation_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("plot saved to orientation_result.png")

print(f"\nPCA        mean offset: {delta_pca.mean():+.1f}°   std: {delta_pca.std():.1f}°")
print(f"Grid search mean offset: {delta_grid.mean():+.1f}°   std: {delta_grid.std():.1f}°")

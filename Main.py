from AngDistObsLSA import *


# read files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
control = pd.read_csv("Control.txt")
obs = pd.read_csv("Observations.csv")
initial_coord = pd.read_csv("Unknown Points Approximate Coordinates.csv")


# remove blank rows ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
control = control.dropna(axis=0, how='all')
control = control.dropna(axis=1, how='all')
obs = obs.dropna(axis=0, how='all')
obs = obs.dropna(axis=1, how='all')
initial_coord = initial_coord.dropna(axis=0, how='all')
initial_coord = initial_coord.dropna(axis=1, how='all')
print('control coordinates:\n')
print(control)
print('\n\n\n')
print('unknown points coordinates: \n')
print(initial_coord)
print('\n\n\n')
print('observations input: \n')
print(obs)
print('\n\n\n')


# Least Square Adjustment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test = AngDistLSA(control, initial_coord, obs, 1.5, 0.02)


# output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sigma = test.sigma_vec()
unk_pt_coor_best_estimation = test.get_unknown_points_coordinate()
residual = test.get_residual()
adj_obs = test.get_adjusted_observations()
err_ellipse = test.plot_standard_error_ellipse()


# a-posteriori factor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
a_post_factor = test.get_a_posteriori_factor()


# covariance matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cov_unk = test.get_covariance_unknown()
cov_l_hat = pd.DataFrame(test.get_covariance_adjusted_observation())
cov_v_hat = pd.DataFrame(test.get_covariance_residual())


# global test and local test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test.global_test()
loc_test_res = test.local_test()
print('local test:\n', loc_test_res)


# run iterative least square adjustment with local test removing potential outlier ~~~~~~~~~~~~~~~
res_after_loc = local_test_and_lsa_after_removal(control, initial_coord, obs, 1.5, 0.02, loc_test_res)


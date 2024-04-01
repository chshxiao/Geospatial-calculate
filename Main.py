from AngDistObsLSA import *


## read files
control = pd.read_csv("Control.txt")
obs = pd.read_csv("Observations.csv")
initial_coord = pd.read_csv("Unknown Points Approximate Coordinates.csv")


## remove blank rows
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

test = AngDistLSA(control, initial_coord, obs, 1.5, 0.02)

# # output
# print('sigma\n', test.sigma_vec)
# print('\n\n\n')
# print('final unknown points coordinates\n', test.get_unknown_points_coordinate())
# print('\n\n\n')
# print('residual:\n', test.get_residual())
# print('\n\n\n')
# print('adjusted observations:\n', test.get_adjusted_observations())
# print('\n\n\n')
# print('standard ellipse of unknown points\n', test.plot_standard_error_ellipse())

# # a-posteriori factor
# print('a_posteriori factor:\n', test.get_a_posteriori_factor())
# print('\n\n\n')

# covariance matrix
print('covariance matrix of unknown:\n', test.get_covariance_unknown())
print('\n\n\n')
cov_l_hat = pd.DataFrame(test.get_covariance_adjusted_observation())
cov_l_hat.to_csv("cov_adj_obs.csv")
cov_v_hat = pd.DataFrame(test.get_covariance_residual())
cov_v_hat.to_csv("cov_res.csv")

# # global test and local test
# test.global_test()
# loc_test_res = test.local_test()
# print('local test:\n', loc_test_res)
# print('\n\n\n')
#
#
# res_after_loc = local_test_and_lsa_after_removal(control, initial_coord, obs, 1.5, 0.02, loc_test_res)
#
# # output
# print('sigma\n', res_after_loc.sigma_vec)
# print('\n\n\n')
# print('final unknown points coordinates\n', res_after_loc.get_unknown_points_coordinate())
# print('\n\n\n')
# print('residual:\n', res_after_loc.get_residual())
# print('\n\n\n')
# print('adjusted observations:\n', res_after_loc.get_adjusted_observations())
# print('\n\n\n')
# print('standard ellipse of unknown points\n', res_after_loc.plot_standard_error_ellipse())
#
# # a-posteriori factor
# print('a_posteriori factor:\n', res_after_loc.get_a_posteriori_factor())
# print('\n\n\n')
#
# # covariance matrix
# print('covariance matrix of unknown:\n', res_after_loc.get_covariance_unknown())
# print('\n\n\n')
# cov_l_hat = pd.DataFrame(res_after_loc.get_covariance_adjusted_observation())
# cov_l_hat.to_csv("cov_adj_obs2.csv")
# cov_v_hat = pd.DataFrame(res_after_loc.get_covariance_residual())
# cov_v_hat.to_csv("cov_res2.csv")

import matplotlib.patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AngleDistanceObservations import *
from scipy.stats import chi2, norm


class AngDistLSA:
    """
    This class is doing Least Square Adjustment on angle and distance observations 2D network.
    It takes pandas dataframes for control points coordinates, unknown points coordinates approximation, observations,
    and the set angle error and distance error. The units of two errors are sec and m respectively.
    Least Square Adjustment process is done once the class is initialize. You can output the results including
    best estimation, covariance, and etc.
    """
    def __init__(self, control, approx, obs, ang_err, dist_err):
        num_obs = obs.shape[0]
        num_unk = approx.shape[0] * 2

        # data
        self.control = control.copy()
        self.approx = approx.copy()
        self.obs = obs.copy()

        # least square adjustment matrices
        self.a_mat = np.zeros((num_obs, num_unk))
        self.cl_mat = self.__create_cl_mat__(self, ang_err, dist_err)
        self.p_mat = np.linalg.inv(self.cl_mat)
        self.n_mat = np.zeros((num_unk, num_unk))
        self.w_vec = np.zeros((num_obs, 1))
        self.sigma_vec = np.zeros((num_unk, 1))

        # output after adjustment
        self.x_hat = np.zeros((num_unk, 1))
        self.residual = np.zeros((num_obs, 1))
        self.l_hat = np.zeros((num_obs, 1))

        # least square adjustment process
        self.__lsa_process__()

        # output
        self.__cal_x_hat_v_hat_l_hat__()

    @staticmethod
    def __create_cl_mat__(self, ang_err, dist_err):
        """
        This function create observation covariance matrix.
        It takes a-priori angle error and distance error and
        assumes they don't change with range
        :param ang_err: constant angle error
        :param dist_err: constant distance error
        """
        ang_err_rad = ang_err/3600*math.pi/180
        cl_mat = np.zeros((self.obs.shape[0], self.obs.shape[0]))
        for i in range(0, self.obs.shape[0]):
            if self.obs.at[i, 'Measure type'] == 'angle':
                cl_mat[i, i] = ang_err_rad**2
            else:
                cl_mat[i, i] = dist_err**2

        return cl_mat

    def __create_a_matrix_and_w_vector__(self):
        """
        This function creates the design A matrix and the misclosure vector w.
        """

        num_obs = self.obs.shape[0]
        for i in range(0, num_obs):

            # angle observations
            if self.obs.at[i, 'Measure type'] == 'angle':
                ang_obs, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
                    self.__ang_obs_data_extraction__(self.obs.iloc[i], i)

                dist_from_at = math.sqrt((ang_obs.from_pt.x - ang_obs.at_pt.x)**2 +
                                         (ang_obs.from_pt.y - ang_obs.at_pt.y)**2)
                dist_to_at = math.sqrt((ang_obs.to_pt.x - ang_obs.at_pt.x)**2 +
                                       (ang_obs.to_pt.y - ang_obs.at_pt.y)**2)

                # calculate angle observations A matrix
                if len(a_mat_iter1) > 0:
                    # From point
                    if iter1_pos == 'From':
                        a_mat_iter1[0, 0] = -(ang_obs.from_pt.y - ang_obs.at_pt.y)/(dist_from_at**2)
                        a_mat_iter1[0, 1] = (ang_obs.from_pt.x - ang_obs.at_pt.x)/(dist_from_at**2)

                    elif iter1_pos == 'At':
                        a_mat_iter1[0, 0] = (ang_obs.from_pt.y - ang_obs.at_pt.y)/(dist_from_at**2) - \
                                            (ang_obs.to_pt.y - ang_obs.at_pt.y)/(dist_to_at**2)
                        a_mat_iter1[0, 1] = (ang_obs.to_pt.x - ang_obs.at_pt.x)/(dist_to_at**2) - \
                                            (ang_obs.from_pt.x - ang_obs.at_pt.x) / (dist_from_at ** 2)

                    elif iter1_pos == 'To':
                        a_mat_iter1[0, 0] = (ang_obs.to_pt.y - ang_obs.at_pt.y)/(dist_to_at**2)
                        a_mat_iter1[0, 1] = -(ang_obs.to_pt.x - ang_obs.at_pt.x)/(dist_to_at**2)

                    if len(a_mat_iter2) > 0:
                        if iter2_pos == 'At':
                            a_mat_iter2[0, 0] = (ang_obs.from_pt.y - ang_obs.at_pt.y) / (dist_from_at ** 2) - \
                                                (ang_obs.to_pt.y - ang_obs.at_pt.y) / (dist_to_at ** 2)
                            a_mat_iter2[0, 1] = (ang_obs.to_pt.x - ang_obs.at_pt.x) / (dist_to_at ** 2) - \
                                                (ang_obs.from_pt.x - ang_obs.at_pt.x) / (dist_from_at ** 2)
                        elif iter2_pos == 'To':
                            a_mat_iter2[0, 0] = (ang_obs.to_pt.y - ang_obs.at_pt.y) / (dist_to_at ** 2)
                            a_mat_iter2[0, 1] = -(ang_obs.to_pt.x - ang_obs.at_pt.x) / (dist_to_at ** 2)

                # calculate angle misclosure
                cal_ang = math.atan2(ang_obs.from_pt.x - ang_obs.at_pt.x,
                                     ang_obs.from_pt.y - ang_obs.at_pt.y) - \
                          math.atan2(ang_obs.to_pt.x - ang_obs.at_pt.x,
                                     ang_obs.to_pt.y - ang_obs.at_pt.y)

                if cal_ang < 0:
                    cal_ang = cal_ang + 2 * math.pi

                self.w_vec[i, 0] = cal_ang - ang_obs.angle_rad

            # distance observation
            elif self.obs.at[i, 'Measure type'] == 'distance':
                dist_obs, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
                    self.__dist_obs_data_extraction__(self.obs.iloc[i], i)

                dist_from_to = math.sqrt((dist_obs.from_pt.x - dist_obs.to_pt.x)**2 +
                                         (dist_obs.from_pt.y - dist_obs.to_pt.y)**2)

                # calculate distance observation A matrix
                if len(a_mat_iter1) > 0:
                    # From point
                    if iter1_pos == 'From':
                        a_mat_iter1[0, 0] = (dist_obs.to_pt.x - dist_obs.from_pt.x) / dist_from_to
                        a_mat_iter1[0, 1] = (dist_obs.to_pt.y - dist_obs.from_pt.y) / dist_from_to

                    elif iter1_pos == 'To':
                        a_mat_iter1[0, 0] = (dist_obs.from_pt.x - dist_obs.to_pt.x) / dist_from_to
                        a_mat_iter1[0, 1] = (dist_obs.from_pt.y - dist_obs.to_pt.y) / dist_from_to

                    if len(a_mat_iter2) > 0:
                        # the second unknown point only in To point
                        a_mat_iter2[0, 0] = (dist_obs.from_pt.x - dist_obs.to_pt.x) / dist_from_to
                        a_mat_iter2[0, 1] = (dist_obs.from_pt.y - dist_obs.to_pt.y) / dist_from_to

                # calculate distance misclosure
                self.w_vec[i, 0] = dist_from_to - dist_obs.dist

    def __lsa_process__(self):
        """
        This function uses the CL matrix, A matrix, and the w vector created
        to find the best estimation of solution. The function keeps iterate until
        the norm of the error vector of unknowns match the specification or the number
        of loop is more than 10.
        """

        max_loop_count = 10
        max_norm = 10**-6

        for loop in range(0, max_loop_count):

            self.__create_a_matrix_and_w_vector__()

            # normal matrix N
            self.n_mat = self.a_mat.transpose() @ self.p_mat @ self.a_mat

            # normal vector u
            u_vec = self.a_mat.transpose() @ self.p_mat @ self.w_vec

            # best estimation of unknown error sigma
            self.sigma_vec = -np.linalg.inv(self.n_mat) @ u_vec

            # not the best estimation yet
            if np.linalg.norm(self.sigma_vec) >= max_norm:
                for i in range(0, self.approx.shape[0]):
                    self.approx.at[i, 'X'] = self.approx.at[i, 'X'] - self.sigma_vec[i*2, 0]
                    self.approx.at[i, 'Y'] = self.approx.at[i, 'Y'] - self.sigma_vec[i*2+1, 0]

            # we got the best estimation
            else:
                break

    def __cal_x_hat_v_hat_l_hat__(self):
        """
        This function calculates the best solution for unknowns, residuals, and observations after lsa
        :return:
        """

        num_unk_pt = self.approx.shape[0]
        num_obs = self.obs.shape[0]

        # calculate best solution for unknowns
        for i in range(0, num_unk_pt):
            self.x_hat[i * 2, 0] = self.approx.at[i, 'X'] + self.sigma_vec[i * 2, 0]
            self.x_hat[i * 2 + 1, 0] = self.approx.at[i, 'Y'] + self.sigma_vec[i * 2 + 1, 0]

        # calculate best solution for observations and residuals
        self.residual = self.a_mat @ self.sigma_vec + self.w_vec
        for i in range(0, num_obs):
            if self.obs.at[i, 'Measure type'] == 'angle':
                self.l_hat[i, 0] =\
                    (self.obs.at[i, 'Value1'] + self.obs.at[i, 'Value2']/60 + self.obs.at[i, 'Value3']/3600) * \
                    math.pi / 180 + self.residual[i, 0]

            elif self.obs.at[i, 'Measure type'] == 'distance':
                self.l_hat[i, 0] =\
                    self.obs.at[i, 'Value1'] + self.residual[i, 0]

    def get_unknown_points_coordinate(self):
        """
        This function returns the best estimation of unknown points coordinate in the format:
        Points      X       Y
        """
        final = self.approx

        for i in range(0, final.shape[0]):
            final.at[i, 'X'] = self.x_hat[i*2, 0]
            final.at[i, 'Y'] = self.x_hat[i*2+1, 0]

        return final

    def get_adjusted_observations(self):
        return self.l_hat

    def get_residual(self):
        return self.residual

    def get_a_posteriori_factor(self):
        """
        This function returns the a-posteriori factor.
        :return:
        """

        # degree of freedom
        dof = self.obs.shape[0] - self.approx.shape[0] * 2

        # a-posteriori factor
        a_post_fac = self.residual.transpose() @ self.p_mat @ self.residual / dof

        return a_post_fac[0, 0]

    def get_covariance_unknown(self):
        return np.linalg.inv(self.n_mat)

    def get_covariance_adjusted_observation(self):
        cx = self.get_covariance_unknown()
        return self.a_mat @ cx @ self.a_mat.transpose()

    def get_covariance_residual(self):
        cl_hat = self.get_covariance_adjusted_observation()
        return self.cl_mat - cl_hat

    def global_test(self, confidence_level=0.95):
        """
        This function does the global test on the network. It compares the a-priori factor and the a-posteriori factor
        and check the fitness of the model with 95% confidence level by default
        (too optimistic/pessimistic/model doesn't fit)
        It uses chi-square distribution from scipy package
        :param: confidence_level: the value to decide whether to accept the hypothesis or not
        :return: 1 or 0. 1 if global test passed. 0 if global test failed.
        """

        # degree of freedom
        dof = self.obs.shape[0] - self.x_hat.shape[0]

        # y value
        y_value = dof * self.get_a_posteriori_factor()

        # chi-square value
        significant_level = [(1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2]
        bounds = chi2.ppf(significant_level, dof)

        if (y_value >= bounds[0]) and (y_value <= bounds[1]):
            print("global test passed")
            return 1
        else:
            print("global test failed")
            return 0

    def local_test(self, confidence_level=0.95):
        """
        This function does the local test on the residual after least square adjustment.
        :return:
        """

        # covariance matrix of residual
        cv_hat = self.get_covariance_residual()

        # normalized residual as y value for the test
        num_obs = self.obs.shape[0]
        res = pd.DataFrame(columns=['ID', 'test_value', 'result'])

        significant_level = [(1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2]
        lb, ub = norm.ppf(significant_level)

        for i in range(0, num_obs):

            y_value = float(self.residual[i] / math.sqrt(cv_hat[i, i]))

            if (y_value >= lb) and (y_value <= ub):
                res.loc[i] = [self.obs.at[i, 'ID'], y_value, "passed"]
            else:
                res.loc[i] = [self.obs.at[i, 'ID'], y_value, "failed"]

        return res

    def plot_standard_error_ellipse(self):
        """
        Output the semi-major-axis and semi-minor-axis with the orientation of the standard ellipse
        of each unknown point. It returns a pandas dataframe with the following format:
        Point      X       Y       a       b       phi
        At the end, the ellipses are plotted
        :return: ellip: pandas Dataframe of ellipses
        """

        # eigen values of covariance matrix for unknowns
        num_unk_pt = self.approx.shape[0]
        cx_hat = self.get_covariance_unknown()
        a_post = self.get_a_posteriori_factor()
        dof = self.obs.shape[0] - self.x_hat.shape[0]
        c_value = chi2.ppf(0.95, dof)

        # dataframe for ellipses
        ellip = self.get_unknown_points_coordinate()

        for i in range(0, num_unk_pt):
            cx_pt = cx_hat[i*2:i*2+2, i*2:i*2+2]
            eigen_val = np.linalg.eigvals(cx_pt)
            ellip.at[i, 'a'] = eigen_val.max() * a_post * c_value
            ellip.at[i, 'b'] = eigen_val.min() * a_post * c_value
            ellip.at[i, 'azi'] = 0.5 * math.atan2(2*cx_pt[0, 1], cx_pt[0, 0] - cx_pt[1, 1])

        # plot the ellipses
        t = np.linspace(0, 2*math.pi, 100)
        for i in range(0, num_unk_pt):
            plt.plot(ellip.at[i, 'X'] + ellip.at[i, 'a'] * np.cos(t),
                     ellip.at[i, 'Y'] + ellip.at[i, 'b'] * np.sin(t))

        plt.show()

        return ellip

    def __ang_obs_data_extraction__(self, obs, ind):
        """
        This function takes the control and unknown point coordinates and an angle observations
        It returns three points coordinates and the reference to the A matrix
        :param obs: an angle observation

        :return: ang_obs: the angle observation class containing three points coordinates and the measurement
        :return: a_mat_iter1: the reference to the position of the first unknown point in observation
        :return: iter1_pos: 'From', 'At', or 'To' string of the first unknown point
        :return: a_mat_iter2: the reference to the positon of the second unknown point in observation
        :return: iter2_pos: 'From', 'At', or 'To' string of the second unknown point
        """

        # initialization
        a_mat_iter1 = []
        iter1_pos = ''
        a_mat_iter2 = []
        iter2_pos = ''

        # control points or unknown points
        obs_points = obs.loc[['From', 'At', 'To']]
        is_cont_obs = obs_points.isin(self.control['Points'].tolist())

        # get the information of three points
        # From point
        from_pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
            self.__find_pt_coor_and_ref2a_mat__(obs, ind, is_cont_obs, 'From',
                                                a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos)

        # At point
        at_pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
            self.__find_pt_coor_and_ref2a_mat__(obs, ind, is_cont_obs, 'At',
                                                a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos)

        # To point
        to_pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
            self.__find_pt_coor_and_ref2a_mat__(obs, ind, is_cont_obs, 'To',
                                                a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos)

        ang_obs = AngleObs(from_pt, at_pt, to_pt, obs.loc[['Value1', 'Value2', 'Value3']].to_numpy())

        return ang_obs, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos

    def __dist_obs_data_extraction__(self, obs, ind):
        """
        This function takes the control and unknown point coordinates and an angle observations
        It returns three points coordinates and the reference to the A matrix
        :param obs: a distance observation

        :return: dist_obs: the angle observation class containing three points coordinates and the measurement
        :return: a_mat_iter1: the reference to the position of the first unknown point in observation
        :return: iter1_pos: 'From', 'At', or 'To' string of the first unknown point
        :return: a_mat_iter2: the reference to the positon of the second unknown point in observation
        :return: iter2_pos: 'From', 'At', or 'To' string of the second unknown point
        """
        # initialization
        a_mat_iter1 = []
        iter1_pos = ''
        a_mat_iter2 = []
        iter2_pos = ''

        # control points or unknown points
        obs_points = obs.loc[['From', 'At', 'To']]
        is_cont_obs = obs_points.isin(self.control['Points'].tolist())

        # get the information of three points
        # From point
        from_pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
            self.__find_pt_coor_and_ref2a_mat__(obs, ind, is_cont_obs, 'From',
                                                a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos)

        # To point
        to_pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos = \
            self.__find_pt_coor_and_ref2a_mat__(obs, ind, is_cont_obs, 'To',
                                                a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos)

        dist_obs = DistObs(from_pt, to_pt, obs.Value1.item())

        return dist_obs, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos

    def __find_pt_coor_and_ref2a_mat__(self, obs, ind, is_cont_obs, pt_pos,
                                       a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos):
        """
        This function does the process of finding coordinate of the point given and
        the reference to the corresponding position in the A matrix if the point is an unknown point
        :param: control: control points coordinates
        :param: approx: unknown points approximated coordinates
        :param: obs: an angle or distance observation
        :param: is_cont_obs: if From, At, To points are control points
        :param: pt_pos: point position in the observation, str

        :return:
        """

        row = ind

        if is_cont_obs.at[pt_pos]:
            cont_pt = self.control.loc[self.control['Points'] == obs.at[pt_pos]]
            pt = Point(cont_pt.Points.item(), cont_pt.X.item(), cont_pt.Y.item())

        else:
            # for unknown point, reference to the a matrix index and the point position in the observation
            unk_pt_index = self.approx.loc[self.approx['Points'] == obs.at[pt_pos]].index[0]
            unk_pt = self.approx.loc[unk_pt_index]
            pt = Point(unk_pt.at['Points'], unk_pt.at['X'], unk_pt['Y'])

            # reference to the corresponding position in A matrix
            if len(a_mat_iter1) == 0:
                a_mat_iter1 = self.a_mat[row:(row + 1), unk_pt_index * 2:(unk_pt_index * 2 + 2)]
                iter1_pos = pt_pos
            else:
                a_mat_iter2 = self.a_mat[row:(row + 1), unk_pt_index * 2:(unk_pt_index * 2 + 2)]
                iter2_pos = pt_pos

        return pt, a_mat_iter1, iter1_pos, a_mat_iter2, iter2_pos


def local_test_and_lsa_after_removal(control, approx, obs, ang_err, dist_err, local_test_res):
    """
    Eliminate the observation with largest normalized residual and do the LSA process again.
    :return:
    """

    new_obs = obs.copy()
    res = local_test_res

    while True:
        # find the largest normalized residual from local test
        abs_res = res.loc[:, 'test_value'].abs()
        removal_index = abs_res[abs_res == abs_res.max()].index[0]

        # remove the observation from original data
        new_obs.drop(index=removal_index, inplace=True)
        new_obs.reset_index(inplace=True, drop=True)

        # create a new AngDistObs class
        new_set = AngDistLSA(control, approx, new_obs, ang_err, dist_err)

        new_set.global_test()
        res = new_set.local_test()
        print(res)

        check = res[res['result'] == 'failed'].shape[0]

        if check == 0:
            break

    return new_set

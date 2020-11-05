from utils import utils as ul
import numpy as np
import matplotlib.pyplot as plt
import pickle
from dipy.data import default_sphere

if __name__ == '__main__':
    sphere = ul.create_sphere(30)
    points = ul.get_points_coordinates(sphere)

    data_test, data_retest = ul.load_images()
    test_img = data_test[0]
    test_bvals = data_test[1]
    test_bvecs = data_test[2]

    retest_img = data_retest[0]
    retest_bvals = data_retest[1]
    retest_bvecs = data_retest[2]
    # bvecs from http://www.emmanuelcaruyer.com/q-space-sampling.php
    bvecs_generated = np.array([
        [0.049, -0.919,	-0.391],
        [0.726, 0.301, -0.618],
        [-0.683, 0.255,	-0.684],
        [0.845,	-0.502,	-0.186],
        [-0.730, -0.619, -0.288],
        [-0.051, 0.039, 0.998],
        [-0.018, 0.871,	-0.491],
        [-0.444, 0.494, 0.747],
        [-0.989, -0.086, -0.116],
        [-0.470, -0.855, 0.221],
        [0.412,	0.400, 0.819],
        [-0.552, 0.790,	-0.267],
        [-0.123, -0.477, 0.871],
        [-0.848, 0.141,	0.510],
        [-0.341, -0.788, -0.512],
        [0.361,	-0.529,	0.768],
        [-0.472, 0.850,	0.234],
        [-0.856, -0.481, 0.189],
        [0.797,	0.162, 0.582],
        [0.467,	-0.009,	-0.884],
        [0.013,	0.998, -0.056],
        [0.882,	-0.387,	0.267],
        [0.017,	-0.536,	-0.844],
        [-0.442, -0.651, 0.617],
        [0.365,	-0.058,	0.929],
        [0.977,	-0.004,	-0.213],
        [-0.406, -0.902, -0.145],
        [-0.627, 0.614, 0.479],
        [-0.354, 0.772,	-0.528],
        [-0.658, -0.472, -0.586],
        [0.423, 0.322, -0.847],
        [0.212,	-0.754,	-0.622],
        [0.912,	-0.104,	0.398],
        [-0.311, 0.947,	-0.077],
        [0.679,	0.632, -0.374],
        [0.135,	-0.286, 0.949],
        [-0.647, 0.230, 0.727],
        [0.904, 0.397, 0.158],
        [-0.757, 0.647, -0.087],
        [0.143, 0.284, 0.948]])

    # create gradient table from generated bvecs
    gtab_generated = ul.create_own_gradient_table(bvecs_generated)
    # prepare data
    first_30_images, next_60_images, grad_table_first_30, grad_table_next_60 = ul.prepare_data(test_img, test_bvecs,
                                                                                           test_bvals)
    # with open('results/masked_30.pickle', 'wb') as file_1:
    #     pickle.dump(ul.remove_background(first_30_images), file_1)
    # with open('results/masked_60.pickle', 'wb') as file_2:
    #     pickle.dump(ul.remove_background(next_60_images), file_2)

    with open('results/masked_30.pickle', 'rb') as file_1:
        masked_30 = pickle.load(file_1)
    with open('results/masked_60.pickle', 'rb') as file_2:
        masked_60 = pickle.load(file_2)

    reference_30 = masked_30[:, :, 39, 15]
    reference_60 = masked_60[:, :, 39, 15]

    # predicting the signal
    transformed_all = ul.rotate_3d(masked_60[:, :, 39:41], 5)
    # untransformed = ul.transform_image(transformed, rotation=-5)
    predicted_30 = ul.predict_image(grad_table_first_30, gtab_generated, masked_30[:, :, 39:40])
    predicted_60 = ul.predict_image(grad_table_next_60, gtab_generated, transformed_all[:, :, 0:1])

    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(predicted_30[:, :, 0, 10], cmap='gray')
    # plt.title("predicted_30")
    # plt.subplot(2,2,2)
    # plt.imshow(predicted_60[:, :, 0, 10], cmap='gray')
    # plt.title("predicted_60")
    # plt.subplot(2,2,3)
    # plt.imshow(reference_30, cmap='gray')
    # plt.title("reference_30")
    # plt.subplot(2,2,4)
    # plt.imshow(transformed_all[:, :, 0, 10], cmap='gray')
    # plt.title("reference_60")
    # plt.show()
    ul.rotate_and_measure(reference_30, transformed_all[:, :, 0, 15])
    ul.rotate_and_measure(predicted_30[:, :, 0, 15], predicted_60[:, :, 0, 15])

    # transformed = ul.rotate_3d(masked_60[:, :, 39:41], 5)
    # odf_60_rotated = ul.quick_fodf(grad_table_next_60, transformed[:, :, 0:1], default_sphere)
    # odf_30 = ul.quick_fodf(grad_table_first_30, masked_30[:, :, 39:40], default_sphere)
    # print("Initial 5 degrees rotation, euclidean distance: {}".format(ul.euclidean_distance(odf_30, odf_60_rotated)))
    # for angle in (range(-10, 10, 1)):
    #     rotated = ul.rotate_3d(transformed, angle)
    #     odf_60 = ul.quick_fodf(grad_table_next_60, rotated[:, :, 0:1], default_sphere)
    #     print("Angle: {}   Euclidean distance: {}".format(angle, ul.euclidean_distance(odf_30, odf_60)))



    #calculate fodfs
    #first 30 images

    # csd_30 = ul.calculate_fodf(grad_table_first_30, masked_30[:, :, 39:40], 'masked_whole_30_sphere', sphere)
    # # next 60 images(undersampled) rotated
    # csd_60 = ul.calculate_fodf(grad_table_next_60, all_images_rotated[:, :, 0:1], 'masked_rotated_whole_60_sphere', sphere)



    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(csd_30.predict(grad_table_first_30)[:, :, 0, 1], cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(csd_60.predict(grad_table_first_30)[:, :, 0, 1], cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(csd_60.predict(grad_table_next_60)[:, :, 0, 1], cmap='gray')
    # plt.subplot(2,2,4)
    # plt.imshow(masked_30[:, :, 39, 1], cmap='gray')
    # plt.show()
    # with open('results/csd_30_fit.pickle', 'wb') as file_1:
    #     pickle.dump(csd_30, file_1)
    # with open('results/csd_60_fit.pickle', 'wb') as file_2:
    #     pickle.dump(csd_60, file_2)
    # # calculate odfs
    # # first 30 images
    # csa_odf_30 = ul.calculate_odf(grad_table_first_30, first_30_images, 8)
    # # next 60 images
    # csa_odf_60 = ul.calculate_odf(grad_table_next_60, next_60_images, 8)
    # print(ul.mutual_information(masked_30[:, :, 39, 1], csd_60.predict(grad_table_first_30)[:, :, 0, 1], 50))
    # print(ul.mutual_information(masked_30[:, :, 39, 1], csd_60.predict(grad_table_next_60)[:, :, 0, 1], 50))

    # calculate q-ball odfs
    # first 30 images
    # qball_odf_30, qball_coeff_30 = ul.qball(grad_table_first_30, masked_30, 'masked_whole_30')
    # # next 60 images
    # qball_odf_60, qball_coeff_60 = ul.qball(grad_table_next_60, next_60_images, 'whole_60')
    # dist = ul.euclidean_distance(qball_odf_30, qball_odf_60)
    # print(dist)



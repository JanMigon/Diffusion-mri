
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# niftii support
import nibabel as nib
import geomstats as gs
from skimage.transform import AffineTransform, warp, SimilarityTransform
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response
from dipy.core.gradients import gradient_table
from scipy import ndimage
from skimage import restoration
from dipy.io import read_bvals_bvecs
from dipy.viz import regtools
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import AffineTransform3D
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.viz import window, actor
from dipy.reconst.shm import CsaOdfModel, QballModel
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges

def load_hcp_data(data_path, bvals_path, bvecs_path):
    # load HCP data
    img = nib.load(data_path)
    data = img.get_data()

    # load b-values and vectors
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
    return data, bvals, bvecs


def load_images():
    # paths for test data
    test_data_path = "resources/data/test/172332/T1w/Diffusion/data.nii"
    test_bvals_path = "resources/data/test/172332/T1w/Diffusion/bvals"
    test_bvecs_path = "resources/data/test/172332/T1w/Diffusion/bvecs"

    # paths for retest data
    retest_data_path = "resources/data/retest/172332/T1w/Diffusion/data.nii"
    retest_bvals_path = "resources/data/retest/172332/T1w/Diffusion/bvals"
    retest_bvecs_path = "resources/data/retest/172332/T1w/Diffusion/bvecs"

    # load test data
    data_test, bvals_test, bvecs_test = load_hcp_data(test_data_path, test_bvals_path, test_bvecs_path)

    # load retest data
    data_retest, bvals_retest, bvecs_retest = load_hcp_data(retest_data_path, retest_bvals_path, retest_bvecs_path)
    return [data_test, bvals_test, bvecs_test], [data_retest, bvals_retest, bvecs_retest]


def build_filters(ksize, sigma, lambd, gamma, psi):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        params = {'ksize':(ksize, ksize), 'sigma':sigma, 'theta':theta, 'lambd':lambd,
                  'gamma':gamma, 'psi':psi, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        #kern /= 1.5*kern.sum()
        filters.append((kern, theta))
    return filters


def build_gabor_filter(sigma, theta, lambd, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    real_gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi / lambd * x_theta + psi)
    imaginary_gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(
        2 * np.pi / lambd * y_theta + psi)

    magnitude_filter = np.sqrt(np.power(real_gb, 2) + np.power(imaginary_gb, 2))
    return real_gb, imaginary_gb, magnitude_filter


def filter_image_scipy(img, g_filter):
    return ndimage.convolve(img, g_filter)


def filter_image(img, filters):
    results = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_32F, kern)
        results.append((fimg, params))
    return results


def restore_image(filtered_image, kernel):
    restored_image = restoration.richardson_lucy(filtered_image, kernel, iterations=30)
    return restored_image


def transform_affine(static, moving):
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = AffineTransform3D()
    params0 = None
    affine = affreg.optimize(static, moving, transform, params0,
                             None, None, None)
    transformed = affine.transform(moving)

    return transformed


def overlay_3d_images(static, transformed):
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_affine_2.png")


def mutual_information(img1, img2, bins=20):
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]
    # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def grid_search_gabor_filters(img1, img2):
    params = []
    psi = 0
    for theta in np.arange(0, np.pi, np.pi/8):
        for sigma in [2,3,4,5]:
            for lambd in [5, 10, 15, 20, 25, 30]:
                for gamma in [0.25, 0.5, 0.75, 1]:
                    _, __, g_filter= build_gabor_filter(sigma, theta, lambd, psi, gamma)
                    filtered_img1 = filter_image_scipy(img1, g_filter)
                    filtered_img2 = filter_image_scipy(img2, g_filter)
                    params.append((mutual_information(filtered_img1, filtered_img2), {'theta': [theta], 'sigma': [sigma],
                                                                                      'lambd': [lambd], 'psi': [psi],
                                                                                      'gamma': [gamma]}))
    return max(params, key=lambda tuple: tuple[0])


def create_dir_if_not_exists(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def build_gabor_filter_bank(name):
    fig = plt.figure(figsize=(8, 8))
    filters = build_filters(31, 10, 5, 0.5, 0)
    for i in range(16):
        fig.add_subplot(4, 4, i + 1, title='Theta = {} '.format(round(np.rad2deg(filters[i][1])), 2)).axis('off')
        plt.imshow(filters[i][0], cmap='gray')
    plt.savefig('results/{}.png'.format(name))
    return filters


def filter_img_bank(image, filters, name):
    fig = plt.figure(figsize=(8, 8))
    filtered_imgs = []
    for i in range(16):
        img = filter_image_scipy(image, filters[i][0])
        filtered_imgs.append(img)
        fig.add_subplot(4, 4, i + 1, title='Theta = {} '.format(round(np.rad2deg(filters[i][1])), 2)).axis('off')
        plt.imshow(img, cmap='gray')
    plt.savefig('results/{}.png'.format(name))
    return filtered_imgs


def calculate_local_energy(image):
    return np.sum(image**2)


def calculate_mean_amplitude(image):
    return np.sum(np.abs(image))


def build_gabor_jet(filtered_static, filtered_moving):
    static_gabor_jet = filtered_static[0] ** 2
    moving_gabor_jet = filtered_moving[0] ** 2

    for i in np.arange(1, len(filtered_static), 1):
        static_gabor_jet += filtered_static[i] ** 2

    for i in np.arange(1, len(filtered_moving), 1):
        moving_gabor_jet += filtered_moving[i] ** 2

    plt.figure()
    plt.subplot(2, 1, 1, title='Static Gabor jet').axis('off')
    plt.imshow(static_gabor_jet, cmap='gray')
    plt.subplot(2, 1, 2, title='Moving Gabor jet').axis('off')
    plt.imshow(moving_gabor_jet, cmap='gray')
    plt.savefig('results/gabor_jets.png')
    return static_gabor_jet, moving_gabor_jet

def prepare_data(img, bvecs, bvals, m=30, n=60):
    # take only images with bvalue near to 1000
    img_bvals_1000 = img[:, :, :, np.argwhere(abs(bvals - 1000) <= 10).ravel()]
    # take only bvecs corresponding to bvalue close to 1000
    bvecs_1000 = bvecs[abs(bvals - 1000) <= 10]
    # group first m images, m bvecs, m bvals with b0 data as first element
    first_m_images = np.concatenate((img[:, :, :, [0]], img_bvals_1000[:, :, :, :m]), axis=3)
    first_m_bvecs = np.concatenate(([bvecs[0, :]], bvecs_1000[:m, :]))
    first_m_bvals = np.concatenate(([bvals[0]], bvals[abs(bvals - 1000) <= 10][:m]))

    # group next n images, n bvecs, n bvals with b0 data as first element
    next_n_bvecs = np.concatenate(([bvecs[0, :]], bvecs_1000[m:m+n, :]))[::2,:]
    next_n_bvals = np.concatenate(([bvals[0]], bvals[abs(bvals - 1000) <= 10][m:n+m]))[::2]
    next_n_images = np.concatenate((img[:, :, :, [0]], img_bvals_1000[:, :, :, m:n+m]), axis=3)[:,:,:,::2]

    # create gradient tables
    grad_table_first_m = gradient_table(first_m_bvals, first_m_bvecs)
    grad_table_next_n = gradient_table(next_n_bvals, next_n_bvecs)

    return first_m_images, next_n_images, grad_table_first_m, grad_table_next_n


def calculate_fodf(gtab, images, name, sphere=default_sphere, radius=10, fa_threshold=0.7):
    response, ratio = auto_response(gtab, images, roi_radius=radius, fa_thr=fa_threshold)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd_model.fit(images)
    csd_odf = csd_fit.odf(sphere)
    fodf_spheres = actor.odf_slicer(csd_odf, sphere=sphere, scale=0.9,
                                       norm=False, colormap='plasma')

    ren = window.Scene()
    ren.add(fodf_spheres)

    print('Saving illustration as csd_odfs_{}.png'.format(name))
    window.record(ren, out_path='results/csd_odfs_{}.png'.format(name), size=(600, 600))
    return csd_fit


def calculate_odf(gtab, data, sh_order=4):
    csamodel = CsaOdfModel(gtab, sh_order)

    data_small = data[30:65, 40:75, 39:40]
    csa_odf = csamodel.fit(data_small).odf(default_sphere)
    csa_odf = np.clip(csa_odf, 0, np.max(csa_odf, -1)[..., None])
    odf_spheres = actor.odf_slicer(csa_odf, sphere=default_sphere, scale=0.9,
                                    norm=False, colormap='plasma')

    ren = window.Scene()
    ren.add(odf_spheres)

    print('Saving illustration as csa_odfs_{}.png'.format(data.shape[-1] - 1))
    window.record(ren, out_path='results/csa_odfs_{}.png'.format(data.shape[-1] - 1), size=(600, 600))
    return csa_odf


def qball(gtab, data, name, sh_order=4):
    qballmodel = QballModel(gtab, sh_order)

    data_small = data[:, :, 39:40]
    qball_fit = qballmodel.fit(data_small)
    qball_odf = qball_fit.odf(default_sphere)
    odf_spheres = actor.odf_slicer(qball_odf, sphere=default_sphere, scale=0.9,
                                   norm=False, colormap='plasma')

    ren = window.Scene()
    ren.add(odf_spheres)

    print('Saving illustration as qball_odfs_{}.png'.format(name))#data.shape[-1] - 1))
    window.record(ren, out_path='results/qball_odfs_{}.png'.format(name), size=(600, 600))
    return qball_odf, qball_fit.shm_coeff


def euclidean_distance(odf_1, odf_2):
    return np.sum(np.sqrt(np.sum(np.sum((odf_1 - odf_2)**2, axis=0), axis=0)), axis=1)


def remove_background(image):
    maskdata, mask = median_otsu(image, vol_idx=range(image.shape[-1]), median_radius=3, numpass=1, autocrop=True,
                                 dilate=2)
    return maskdata


def show_data(images):
    fig, axs = plt.subplots(3, 3, figsize=(15, 6))
    # fig.subplots_adjust(hspace=.5, wspace=.001)

    for i, ax in enumerate(axs.ravel()):
        ax.imshow(images[:, :, 48, i + 1], cmap='gray')
        ax.set_axis_off()
    plt.show()


def create_sphere(n_pts):
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, _ = disperse_charges(hsph_initial, 5000)
    sph = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))
    return sph

def transform_image(image, rotation=0):
    transform = AffineTransform(rotation=np.deg2rad(rotation))
    shift_y, shift_x = (np.array(image.shape) - 1) / 2.
    shift_fwd = SimilarityTransform(translation=[-shift_x, -shift_y])
    shift_back = SimilarityTransform(translation=[shift_x, shift_y])
    transformed = warp(image, (shift_fwd + ( transform + shift_back )).inverse, order=1, preserve_range=True, mode='constant')
    return transformed

def show_transform(image_1, image_2, image_3):
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(image_1, cmap='gray')
    plt.subplot(3,1,2)
    plt.imshow(image_2, cmap='gray')
    plt.subplot(3, 1, 3)
    plt.imshow(image_3, cmap='gray')
    plt.show()


def rotate_3d(images, rotation):
    all_images_rotated = np.ones((images.shape))
    for z in range(2):
        for gradient in range(images.shape[-1]):
            all_images_rotated[:, :, z, gradient] = transform_image(images[:, :, z, gradient], rotation)
    return all_images_rotated


def calculate_Frechet_mean(points, weights):
    metric = gs.riemannian_metric.RiemannianMetric(dimension=3)
    frechet_mean = metric.mean(points, weights)
    return frechet_mean


def get_points_coordinates(sphere):
    points = np.array([sphere.x, sphere.y, sphere.z]).T
    return points


def quick_fodf(gtab, images, sphere=default_sphere, radius=10, fa_threshold=0.7):
    response, ratio = auto_response(gtab, images, roi_radius=radius, fa_thr=fa_threshold)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd_model.fit(images)
    csd_odf = csd_fit.odf(sphere)
    return csd_odf


def predict_image(gtab_odf, gtab_predict, images, radius=10, fa_threshold=0.7):
    response, ratio = auto_response(gtab_odf, images, roi_radius=radius, fa_thr=fa_threshold)

    csd_model = ConstrainedSphericalDeconvModel(gtab_odf, response)
    csd_fit = csd_model.fit(images)
    return csd_fit.predict(gtab_predict)


def rotate_and_measure(static, moving, angle_range=9):
    for i, angle in enumerate(range(-angle_range, 0, 1)):
        rotated = transform_image(moving, angle)
        overlay_images(static, rotated, 'static', 'mutual_information: {}'.format(np.round(mutual_information(static,
                                                                                                     rotated, 30), 4)),
                       'angle {}'.format(angle) )
        plt.show()


def _tile_plot(imgs, titles, **kwargs):
    """
    Helper function
    """
    # Create a new figure and plot the three images
    fig, ax = plt.subplots(1, len(imgs), figsize=(12,8))
    for ii, a in enumerate(ax):
        a.set_axis_off()
        a.imshow(imgs[ii], **kwargs)
        a.set_title(titles[ii])

    return fig


def overlay_images(img0, img1, title0='', title_mid='', title1='', fname=None):
    r""" Plot two images one on top of the other using red and green channels.
    Creates a figure containing three images: the first image to the left
    plotted on the red channel of a color image, the second to the right
    plotted on the green channel of a color image and the two given images on
    top of each other using the red channel for the first image and the green
    channel for the second one. It is assumed that both images have the same
    shape. The intended use of this function is to visually assess the quality
    of a registration result.
    Parameters
    ----------
    img0 : array, shape(R, C)
        the image to be plotted on the red channel, to the left of the figure
    img1 : array, shape(R, C)
        the image to be plotted on the green channel, to the right of the
        figure
    title0 : string (optional)
        the title to be written on top of the image to the left. By default, no
        title is displayed.
    title_mid : string (optional)
        the title to be written on top of the middle image. By default, no
        title is displayed.
    title1 : string (optional)
        the title to be written on top of the image to the right. By default,
        no title is displayed.
    fname : string (optional)
        the file name to write the resulting figure. If None (default), the
        image is not saved.
    """
    # Normalize the input images to [0,255]
    img0 = 255 * ((img0 - img0.min()) / (img0.max() - img0.min()))
    img1 = 255 * ((img1 - img1.min()) / (img1.max() - img1.min()))

    # Create the color images
    img0_red = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    img1_green = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    overlay = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)

    # Copy the normalized intensities into the appropriate channels of the
    # color images
    img0_red[..., 0] = img0
    img1_green[..., 1] = img1
    overlay[..., 0] = img0
    overlay[..., 1] = img1

    fig = _tile_plot([img0_red, overlay, img1_green],
                     [title0, title_mid, title1])

    # If a file name was given, save the figure
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    return fig


def create_own_gradient_table(bvecs):
    return gradient_table(1000 * np.ones(bvecs.shape[0]), bvecs)
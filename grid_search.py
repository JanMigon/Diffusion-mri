import pandas as pd
from utils.utils import grid_search_gabor_filters, load_images, create_dir_if_not_exists


if __name__ == '__main__':
    # using gridsearch to find the best gabor filter parameters(psi = const = 0)
    data_test, data_retest = load_images()

    static_test = data_test[:, :, 52, 1]
    moving_test = data_test[:, :, 52, 2]

    max_mi_test = grid_search_gabor_filters(static_test, moving_test)
    params_test = max_mi_test[1]
    df_test = pd.DataFrame(params_test)

    static_retest = data_retest[:, :, 52, 1]
    moving_retest = data_retest[:, :, 52, 1]

    max_mi_retest = grid_search_gabor_filters(static_retest, moving_retest)
    params_retest = max_mi_retest[1]
    df_retest = pd.DataFrame(params_retest)

    params_test_path = "resources/data/test/gabor_filter"
    params_retest_path = "resources/data/retest/gabor_filter"

    create_dir_if_not_exists(params_test_path)
    create_dir_if_not_exists(params_retest_path)

    df_test.to_csv(params_test_path + '/gabor_filter.csv', index=None, header=True)
    df_retest.to_csv(params_retest_path + '/gabor_filter.csv', index=None, header=True)
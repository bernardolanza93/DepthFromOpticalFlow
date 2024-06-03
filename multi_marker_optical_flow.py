from multi_marker_source_function import *




@log_function_call
def find_signal_boundaries(signal, threshold=0.1):
    """
    Find the start and end times of a signal that undergoes rapid changes.

    Parameters:
    signal (np.array): The input signal.
    threshold (float): The threshold for identifying significant changes in the signal.

    Returns:
    (int, int): Start and end times of the significant signal.
    """
    # Calculate the derivative of the signal
    derivative = np.diff(signal)

    # Identify indices where the change is significant
    significant_changes = np.where(np.abs(derivative) > threshold)[0]

    if len(significant_changes) == 0:
        return None, None

    # Get the first and last significant change
    start_idx = significant_changes[0]
    end_idx = significant_changes[-1]

    # Find the real start before the first significant change
    for i in range(start_idx, 0, -1):
        if np.abs(signal[i]) < threshold:
            start_idx = i
            break

    # Find the real end after the last significant change
    for i in range(end_idx, len(signal) - 1):
        if np.abs(signal[i]) < threshold:
            end_idx = i
            break

    return start_idx, end_idx

@log_function_call
def calculate_theo_model_and_analyze(n_df, n_vy, win, vlim):
    """
    Calculate the theoretical model and analyze the data.

    Parameters:
    n_df (DataFrame): The input DataFrame containing data.
    n_vy (array): Array of velocities.
    win (int): Window size for smoothing.
    vlim (float): Velocity limit.

    Returns:
    tuple: Arrays of all_z and all_distance.
    """
    # Separate columns based on suffix '_vx' and '_z'
    vx_columns = [col for col in n_df.columns if '_vx' in col]
    z_columns = [col for col in n_df.columns if '_z' in col]

    # Vectors to collect data
    all_z = []
    all_vx = []
    all_distance = []
    all_vy_robot = []

    for vx_col, z_col in zip(vx_columns, z_columns):
        # Smooth the data
        smoothed_vx = n_df[vx_col].rolling(window=win).mean()
        n_df[vx_col] = smoothed_vx

        dist = []
        for i in range(len(n_df)):
            if n_vy[i] > vlim:
                if n_df[vx_col].iloc[i] != 0:
                    dist.append(n_vy[i] * fx / (n_df[vx_col].iloc[i] * 60))
                else:
                    dist.append(np.nan)  # Handle division by zero

                # Append values to overall vectors
                all_z.append(n_df[z_col].iloc[i])
                all_vx.append(n_df[vx_col].iloc[i] * 60)
                all_distance.append(n_vy[i] * fx / (n_df[vx_col].iloc[i] * 60))
                all_vy_robot.append(n_vy[i])

    # Convert lists to numpy arrays
    all_z = np.array(all_z)
    all_distance = np.array(all_distance)

    # Remove NaN values
    mask = ~np.isnan(all_z) & ~np.isnan(all_distance)
    all_z = all_z[mask]
    all_distance = all_distance[mask]

    # Fit the model Y = X
    y_pred = all_z  # Model Y = X predicts y_pred is equal to x

    # Calculate R^2
    r2 = r2_score(all_distance, y_pred)

    # Calculate other goodness-of-fit metrics
    mse = mean_squared_error(all_distance, y_pred)
    mae = mean_absolute_error(all_distance, y_pred)
    rmse = np.sqrt(mse)

    # Calculate mean dispersion
    mean_dispersion = np.mean(np.abs(all_distance - y_pred))

    print("R^2:", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Dispersion:", mean_dispersion)

    return all_z, all_distance


@log_function_call
def merge_dataset_extr_int(x, vy):
    """
    Merge dataset and extract interesting intervals.

    Parameters:
    x (array): Array of x values.
    vy (array): Array of velocities.

    Returns:
    None
    """
    vy = abs(vy)

    # Path to the CSV file
    file_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/of_raw_re_output_1.csv'

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Remove columns that contain "164"
    df = df.loc[:, ~df.columns.str.contains('164')]

    # Remove columns with zero non-null values
    df = df.dropna(axis=1, how='all')

    # Filter data up to the 3300th record
    df = df.loc[:3299]

    # Generate a timestamp if the timestamp column has no valid data
    df['timestamp'] = range(len(df))
    t = df['timestamp']

    x_vy = x
    FIND_SHIFTER = 0

    if FIND_SHIFTER:
        mean_diff_s_robot = []
        mean_diff_s_cam = []
        meanss = []
        for shift_rob in np.arange(0, 0.006, 0.0001):
            for shift_cam in np.arange(0, 4, 0.1):
                ini, endi = find_signal_boundaries(df["11_vx"], shift_cam)
                ini_rob, endi_rob = find_signal_boundaries(vy, shift_rob)

                if ini is not None and endi is not None:
                    # Trim all signals in the DataFrame using these indices
                    n_df = df.iloc[ini - 2:endi + 3].reset_index(drop=True)

                if ini_rob is not None and endi_rob is not None:
                    n_vy = vy[ini_rob:endi_rob + 1]
                    n_x_vy = x_vy[ini_rob:endi_rob + 1]

                n_df['timestamp'] = n_df['timestamp'] - n_df['timestamp'].iloc[0]
                n_x_vy = n_x_vy - n_x_vy[0]

                # Method 1: Using slicing
                factor = len(n_vy) // len(n_df["11_vx"])

                # Create a new time axis for the decimated signal
                x_vy_decimated = np.linspace(n_df["timestamp"].iloc[0], n_df["timestamp"].iloc[-1], len(n_df["11_vx"]))

                # Interpolate the n_vy signal on the new time axis
                interpolator = interp1d(np.linspace(0, len(n_vy) - 1, len(n_vy)), n_vy, kind='linear')
                n_vy_decimated = interpolator(np.linspace(0, len(n_vy) - 1, len(n_df["11_vx"])))
                n_vy = n_vy_decimated
                n_x_vy = x_vy_decimated

                _11_esay = n_df["11_vx"].tolist()
                massimo_segnale2 = max(_11_esay)
                max1 = max(n_vy)
                segnale_normalizzato = n_vy / max1 * massimo_segnale2

                pointwise_difference = [abs(a - b) for a, b in zip(segnale_normalizzato, _11_esay)]
                data_array = np.array(pointwise_difference)
                mean_difference = np.nanmean(data_array)

                meanss.append(mean_difference)
                mean_diff_s_robot.append(shift_rob)
                mean_diff_s_cam.append(shift_cam)
                print(mean_difference)

        range_list = np.linspace(0, len(meanss) - 1, len(meanss), dtype=int)

        plt.scatter(range_list, meanss)
        plt.show()

        data_array = np.array(meanss)
        min_index = np.nanargmin(data_array)
        thres_robot = mean_diff_s_robot[min_index]
        thres_cam = mean_diff_s_cam[min_index]
        print(min_index, thres_robot, thres_cam)
    else:
        thres_robot = 0.0035
        thres_cam = 1.0

    ini, endi = find_signal_boundaries(df["11_vx"], thres_cam)
    ini_rob, endi_rob = find_signal_boundaries(vy, thres_robot)

    if ini is not None and endi is not None:
        # Trim all signals in the DataFrame using these indices
        n_df = df.iloc[ini - 2:endi + 3].reset_index(drop=True)

    if ini_rob is not None and endi_rob is not None:
        n_vy = vy[ini_rob:endi_rob + 1]
        n_x_vy = x_vy[ini_rob:endi_rob + 1]

    n_df['timestamp'] = n_df['timestamp'] - n_df['timestamp'].iloc[0]
    n_x_vy = n_x_vy - n_x_vy[0]

    factor = len(n_vy) // len(n_df["11_vx"])

    x_vy_decimated = np.linspace(n_df["timestamp"].iloc[0], n_df["timestamp"].iloc[-1], len(n_df["11_vx"]))
    interpolator = interp1d(np.linspace(0, len(n_vy) - 1, len(n_vy)), n_vy, kind='linear')
    n_vy_decimated = interpolator(np.linspace(0, len(n_vy) - 1, len(n_df["11_vx"])))
    n_vy = n_vy_decimated
    n_x_vy = x_vy_decimated

    _11_esay = n_df["11_vx"].tolist()
    massimo_segnale2 = max(_11_esay)
    max1 = max(n_vy)
    segnale_normalizzato = n_vy / max1 * massimo_segnale2

    n_vx_columns = [col for col in n_df.columns if '_vx' in col]
    n_z_columns = [col for col in n_df.columns if '_z' in col]

    # Create a plot with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # First subplot for vx_columns
    for col in n_vx_columns:
        axs[0].plot(n_df['timestamp'], n_df[col], label=col)
    axs[0].plot(n_df['timestamp'], segnale_normalizzato, label="ROB")
    axs[0].set_xlabel('Timestamp')
    axs[0].set_ylabel('VX Values')
    axs[0].set_title('Original VX Signals')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot for vy
    axs[1].plot(n_x_vy, n_vy, label='ROBOT', linestyle='--', color='black')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity (vy)')
    axs[1].set_title('Ground Truth (vy)')
    axs[1].legend()
    axs[1].grid(True)

    # Third subplot for z_columns
    for col in n_z_columns:
        axs[2].plot(n_df['timestamp'], n_df[col], label=col)
    axs[2].set_xlabel('Timestamp')
    axs[2].set_ylabel('Z Values')
    axs[2].set_title('Original Z Signals')
    axs[2].legend()
    axs[2].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

    all_z, all_distance = calculate_theo_model_and_analyze(n_df, n_vy, 1, 0.2)
    all_z_3, all_distance_3 = calculate_theo_model_and_analyze(n_df, n_vy, 5, 0.23)

    # Plot relationships
    plt.figure(figsize=(10, 5))
    plt.scatter(all_z, all_distance, label='Distance vs VX', alpha=0.5, s=1)
    plt.scatter(all_z_3, all_distance_3, label='Distance vs VX mean 3', alpha=0.5, s=1)
    min_val = min(min(all_z), min(all_distance))
    max_val = max(max(all_z), max(all_distance))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y = x)')

    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title('Relationship between VX, Distance, and Z')
    plt.legend()
    plt.grid(True)
    plt.show()


def modello(x, costante):
    """
    The model function for curve fitting.

    Parameters:
    x (float): The independent variable.
    costante (float): The constant for the model.

    Returns:
    float: The dependent variable calculated as costante / x.
    """
    return costante / x


def compute_dz(Vx, Vx_prime, fx, fy, cx, cy):
    """
    Compute the change in depth (dz) based on optical flow.

    Parameters:
    Vx (float): Velocity in x direction.
    Vx_prime (float): Derivative of velocity in x direction.
    fx (float): Focal length in x direction.
    fy (float): Focal length
    """
    dz = ((Vx * fx) / Vx_prime)

@log_function_call
def windowing_vs_uncertanty(file_path):
    """
    Analyze the effect of windowing on the uncertainty of model parameters.

    Parameters:
    file_path (str): The path to the Excel file containing the data.

    Returns:
    None
    """
    SHOW_PLOT = 0

    v_ext = []
    unc_k = []
    sigma_gauss = []
    win_size = []

    for window_size in range(1, 10, 1):
        # Delete the constant file if it exists
        if os.path.exists("constant.txt"):
            os.remove("constant.txt")

        # Create the constant file with headers
        with open("constant.txt", 'w') as file:
            file.write("constant,constant_uncert,velocity\n")

        data = pd.read_excel(file_path)
        data['vx'] = abs(data['vx'])

        # Remove rows with zero or missing values
        data = data[(data != 0).all(1)]

        # Split the DataFrame based on the 'vx_3D' column
        groups = data.groupby('vx_3D')

        # Create a dictionary of sub-dataframes, where each key is a unique value of 'vx_3D'
        sub_dataframes = {key: groups.get_group(key) for key in groups.groups}

        for key, value in sub_dataframes.items():
            data = sub_dataframes[key]

            # Define colors for different values of 'vx_3D'
            color_map = {
                v1: 'red',
                v2: 'azure',
                v3: 'green',
                v4: 'orange',
                v5: 'purple'
            }

            x_fps = data['vx']
            marker_n = data['marker']
            x = [element * 60 for element in x_fps]
            y = data['z_mean']

            SMOOTHING = 1
            window = 0

            if SMOOTHING:
                window = 7
                x_or = x
                x_s = smoothing(x, marker_n, window_size)
                x_s_graph = [x_ii + 1000 for x_ii in x_s]
                x = x_s

            color_p = color_map[key]

            PLOT_OF_RAW = 1
            if PLOT_OF_RAW and SMOOTHING:
                x__1 = list(range(len(x)))
                plt.plot(x__1, x_or)
                plt.plot(x__1, x_s_graph)
                marker_aug = [element * 100 for element in marker_n]
                plt.plot(x__1, marker_aug)
                if SHOW_PLOT:
                    plt.show()

            Vx_prime_values = sorted(x)

            # Fit the model to the data
            params, cov = curve_fit(modello, x, y)

            # Extract the estimated constant
            estimated_constant = params[0]

            # Calculate the uncertainty associated with the constant
            constant_uncertainty = np.sqrt(np.diag(cov))[0]

            # Calculate R^2
            residuals = y - modello(x, estimated_constant)
            residual_sum_of_squares = np.sum(residuals ** 2)
            total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

            # Calculate the model values for plotting
            x_model = np.linspace(min(x), max(x), 100)
            y_model = modello(x_model, estimated_constant)

            # Save the data to the file
            save_to_file_OF_results("constant.txt", estimated_constant, constant_uncertainty, key)

            plt.figure(figsize=(15, 10))

            # Plot raw data points and the model
            plt.scatter(x, y, label='Raw Data', color=color_p, s=35, alpha=0.05, marker="o", edgecolor="black")

            # Generic model plot
            plt.plot(x_model, y_model, label='Generic Model Dz = k/OF', color='black', linestyle='-.')

            plt.xlabel('OF [px/s]')
            plt.ylabel('Depth [m]')
            plt.grid(True)
            plt.ylim(0, 2.1)

            # Additional plot
            Y_theoretical = []
            for i in range(len(Vx_prime_values)):
                dzi = compute_dz(float(key), Vx_prime_values[i], fx, fy, cx, cy)
                Y_theoretical.append(dzi)

            plt.plot(Vx_prime_values, Y_theoretical, color="grey", label='Theoretical Model Dz = (V_r * fx)/OF')

            # Calculate systematic error
            residuals = (y - Y_theoretical) / y
            systematic_error = np.mean(residuals)

            # Calculate random error
            random_error = np.std(residuals)

            theoretical_constant = fx * float(key)

            plt.title(
                f'depth vs Optical flow [z = k / vx] - moving average filter: {window}, \n K_th: {theoretical_constant:.2f} , K_exp: {estimated_constant:.2f} +- {constant_uncertainty:.2f} [px*m] or [px * m/s] || R^2: {r_squared:.4f} \n Stat on relative residuals (asymptotic - non-gaussian): \n epsilon_system_REL: {systematic_error*100 :.3f}% , sigma_REL: {random_error*100 :.3f} %')

            # Position the legend at the top right
            plt.legend(loc="upper right")

            if SHOW_PLOT:
                plt.show()

            if SHOW_PLOT:
                hist_adv(residuals)

            v_ext.append(color_p)
            unc_k.append(constant_uncertainty)
            sigma_gauss.append(random_error)
            win_size.append(window_size)

    plt.close('all')

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot 1: Uncertainty associated with model parameters
    for i in range(len(v_ext)):
        ax1.scatter(win_size[i], unc_k[i], color=v_ext[i], marker="x", label='Model ' + str(i + 1))
    ax1.set_xlabel('Window Size [samples]')
    ax1.set_ylabel('k uncertainty [m*px]')

    # Plot 2: Sigma of the fitted model (Gaussian Sigma)
    for i in range(len(v_ext)):
        ax2.scatter(win_size[i], sigma_gauss[i], color=v_ext[i], label='Model ' + str(i + 1))
    ax2.set_xlabel('Window Size [samples]')
    ax2.set_ylabel('Relative sigma of residuals [std]')

    # Set the title of the subplot
    fig.suptitle('Model Evaluation - Moving Average Effect')

    # Show the plot
    plt.show()

def calculate_distance_vector(x1, y1, x2_array, y2_array):
    """
    Calculate the Euclidean distance between a point and an array of points.

    Parameters:
    x1 (float): x-coordinate of the point.
    y1 (float): y-coordinate of the point.
    x2_array (np.array): Array of x-coordinates of the points.
    y2_array (np.array): Array of y-coordinates of the points.

    Returns:
    np.array: Array of distances.
    """
    return np.sqrt((x1 - x2_array) ** 2 + (y1 - y2_array) ** 2)


@log_function_call
def show_result_ex_file(file_path):
    """
    Show and analyze the results from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file containing the data.

    Returns:
    None
    """
    SHOW_PLOT = 1

    # Delete the constant file if it exists
    if os.path.exists("constant.txt"):
        os.remove("constant.txt")

    # Create the constant file with headers
    with open("constant.txt", 'w') as file:
        file.write("constant,constant_uncert,velocity\n")

    data = pd.read_excel(file_path)
    data['vx'] = abs(data['vx'])

    # Remove rows with zero or missing values
    data = data[(data != 0).all(1)]

    # Split the DataFrame based on the 'vx_3D' column
    groups = data.groupby('vx_3D')

    # Create a dictionary of sub-dataframes, where each key is a unique value of 'vx_3D'
    sub_dataframes = {key: groups.get_group(key) for key in groups.groups}

    for key, value in sub_dataframes.items():
        data = sub_dataframes[key]

        # Define colors for different values of 'vx_3D'
        color_map = {
            v1: 'red',
            v2: 'cyan',
            v3: 'green',
            v4: 'orange',
            v5: 'purple'
        }

        x_fps = data['vx']
        marker_n = data['marker']
        x = [element * 60 for element in x_fps]
        y = data['z_mean']

        SMOOTHING = 0
        window = 0

        if SMOOTHING:
            window = 3
            x_or = x
            x_s = smoothing(x, marker_n, window)
            x_s_graph = [x_ii + 1000 for x_ii in x_s]
            x = x_s

        color_p = color_map[key]

        PLOT_OF_RAW = 1
        if PLOT_OF_RAW and SMOOTHING:
            x__1 = list(range(len(x)))
            plt.plot(x__1, x_or)
            plt.plot(x__1, x_s_graph)
            marker_aug = [element * 100 for element in marker_n]
            plt.plot(x__1, marker_aug)
            if SHOW_PLOT:
                plt.show()

        Vx_prime_values = sorted(x)

        # Fit the model to the data
        params, cov = curve_fit(modello, x, y)

        # Extract the estimated constant
        estimated_constant = params[0]

        # Calculate the uncertainty associated with the constant
        constant_uncertainty = np.sqrt(np.diag(cov))[0]

        # Calculate R^2
        residuals = y - modello(x, estimated_constant)
        residual_sum_of_squares = np.sum(residuals ** 2)
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # Calculate the model values for plotting
        x_model = np.linspace(min(x), max(x), len(x))
        y_model = modello(x_model, estimated_constant)

        # Save the data to the file
        save_to_file_OF_results("constant.txt", estimated_constant, constant_uncertainty, key)

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Additional plot
        Y_theoretical = []
        for i in range(len(Vx_prime_values)):
            dzi = compute_dz(float(key), Vx_prime_values[i], fx, fy, cx, cy)
            Y_theoretical.append(dzi)

        # Calculate the minimum distance for each experimental point
        min_distances = []
        data = pd.DataFrame({
            'x': x,
            'y': y
        })
        model_df = pd.DataFrame({
            'x_model': x_model,
            'y_model': y_model
        })

        for index, row in data.iterrows():
            distances = calculate_distance_vector(row['x'], row['y'], model_df['x_model'].values,
                                                  model_df['y_model'].values)
            min_distance = np.min(distances)
            min_distances.append(min_distance)

        data['min_distance'] = min_distances

        # Display the results
        print(data[['x', 'y', 'min_distance']])

        # Calculate the mean absolute error (MAE)
        mae = np.mean(min_distances)
        print(f'Mean Absolute Error: {mae:.4f}')

        # Prepare the data
        data = pd.DataFrame({
            'x': x,
            'y': y,
            'x_model': x_model,
            'y_model': y_model,
            'Vx_prime_values': Vx_prime_values,
            'Y_theoretical': Y_theoretical
        })

        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'Nimbus Sans',
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.titleweight': 'bold',
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        # Plot raw data points and the model
        sns.scatterplot(x='x', y='y', data=data, label='Raw Data', color=color_p, s=50, alpha=0.7, marker="^",
                        edgecolor="black")
        sns.lineplot(x='x_model', y='y_model', data=data, label=r'Experimental model $d = k_{{exp}}/v_{{px}}$',
                     color='black', linestyle='-.')
        sns.lineplot(x='Vx_prime_values', y='Y_theoretical', data=data, color="grey",
                     label=r'Analytical model $d = V_{{ext}} ⋅ f_{{y}}/v_{{px}}$', alpha=0.7, linewidth=2)

        plt.xlabel(r'$v_{px}$ [$px/s$]', fontsize=16)
        plt.ylabel('Depth [m]', fontsize=16)
        plt.ylim(0, 2.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Calculate systematic error
        residuals = (data['y'] - data['Y_theoretical']) / data['y']
        systematic_error = np.mean(residuals)

        # Calculate random error
        random_error = np.std(residuals)

        theoretical_constant = fx * float(key)

        # Plot title
        plt.title(f'Optical pixel displacement vs. depth. Performed at $V_{{ext}}$ = {key} m/s', fontsize=18,
                  fontweight='bold', pad=15)

        # Position the legend at the top right
        plt.legend(loc="upper right", fontsize=14)

        # Path to save the file
        file_path_fig = 'results/speed_' + str(key) + '_k_model.png'

        # Check if the file already exists
        if os.path.exists(file_path_fig):
            # If the file exists, delete it
            os.remove(file_path_fig)
            print("Removed old plot")

        # Save the figure
        plt.savefig(file_path_fig, dpi=300, bbox_inches='tight')

        if SHOW_PLOT:
            plt.show()


@log_function_call
def constant_analisis():
    """
    Analyze the constant values extracted from the model evaluation at different reference external speeds.

    Parameters:
    None

    Returns:
    None
    """
    # Read data from the file
    data = np.loadtxt("constant.txt", delimiter=',', skiprows=1)

    # Extract columns
    constant_data = data[:, 0]
    constant_uncert_data = data[:, 1]
    velocity_data = data[:, 2]

    # Perform linear regression considering the uncertainty on the constant
    slope, intercept, r_squared = weighted_linregress_with_error_on_y(velocity_data, constant_data,
                                                                      1 / constant_uncert_data)

    # Calculate the uncertainty of the slope
    residuals = constant_data - (slope * velocity_data + intercept)
    uncert_slope = np.sqrt(
        np.sum(constant_uncert_data ** 2 * residuals ** 2) / np.sum((velocity_data - np.mean(velocity_data)) ** 2))

    sigma3 = [element * 3 for element in constant_uncert_data]

    plt.figure(figsize=(12, 7))

    # Plot
    plt.scatter(velocity_data, constant_data, label='Data', s=15)
    plt.errorbar(velocity_data, constant_data, yerr=sigma3, fmt='none', label='Uncertainty')
    plt.plot(velocity_data, slope * velocity_data + intercept, color='red', label='Experimental k(v_ext)')
    plt.plot(velocity_data, velocity_data * fx, color='orange', label='Theoretical K(v_ext)')
    plt.xlabel('V_ext [m/s]')
    plt.ylabel('Constant [k]')
    plt.title(
        f'k_i = f(v_ext) : slope: {slope:.1f} sigma: {uncert_slope:.1f} k/[m/s] || R^2: {r_squared:.4f} \nUncertainty on parameters: {constant_uncert_data[0]:.2f}, {constant_uncert_data[1]:.2f}, {constant_uncert_data[2]:.2f}, {constant_uncert_data[3]:.2f} [px*m] - 99.7% int')
    plt.legend()
    plt.grid(True)

    # Path to save the figure
    file_path_fig = 'results/k_LR.png'

    # Check if the file already exists
    if os.path.exists(file_path_fig):
        # If the file exists, delete it
        os.remove(file_path_fig)
        print("Removed old plot")

    # Save the figure
    plt.savefig(file_path_fig)
    plt.show()

def interpolate_signal(signal_ref, timestamps_ref, signal_other, timestamps_other):
    """
    Interpolate a signal to match the reference timestamps.

    Parameters:
    signal_ref (np.array): Reference signal.
    timestamps_ref (np.array): Timestamps for the reference signal.
    signal_other (np.array): Signal to be interpolated.
    timestamps_other (np.array): Timestamps for the signal to be interpolated.

    Returns:
    np.array: Interpolated signal.
    """
    # Interpolate the signal
    interpolated_signal_other = np.interp(timestamps_ref, timestamps_other, signal_other)
    return interpolated_signal_other

def plot_increment(file_path, label):
    """
    Plot the increment of translation in the X direction normalized to meters per second.

    Parameters:
    file_path (str): The path to the CSV file containing the data.
    label (str): The label for the plot.

    Returns:
    None
    """
    # Read the data from the CSV file
    data = pd.read_csv(file_path, delimiter=",")

    # Extract the minimum timestamp
    min_timestamp = data['__time'].min()

    # Calculate the timestamps relative to zero
    timestamps = data['__time'] - min_timestamp

    # Extract the data for the translation in the X direction
    translation_x = data['/tf/base/tool0_controller/translation/x']

    # Calculate the increment of the translation in the X direction
    translation_increment = translation_x.diff()

    # Calculate the time interval between points
    time_diff = timestamps.diff()

    # Calculate the velocity in meters per second
    velocity_mps = translation_increment / time_diff

    # Plot the increment of translation in the X direction normalized to meters per second
    plt.plot(timestamps[1:], velocity_mps[1:], label=label)

def iter_mp4_files(directory):
    """
    Iterate through all files and directories in the specified directory to find MP4 files.

    Parameters:
    directory (str): The path to the directory to search for MP4 files.

    Yields:
    str: The full path to each MP4 file found.
    """
    # Iterate through all files and directories in the specified directory
    for root, dirs, files in os.walk(directory):
        # Iterate through all files
        for file in files:
            # Check if the file has an MP4 extension
            if file.endswith('.MP4'):
                # Return the full path of the MP4 file
                print(os.path.join(root, file))
                yield os.path.join(root, file)

@log_function_call
def convert_position_to_speed():
    """
    Convert position data to speed data by calculating the increments in 'vx' columns from CSV files.

    Parameters:
    None

    Returns:
    None
    """
    # Get the current directory
    current_directory = os.getcwd()

    # Find all files starting with "raw_re_output_" in the current directory
    matching_files = [file for file in os.listdir(current_directory) if file.startswith("raw_re_output_")]

    # Iterate through each file found
    for file in matching_files:
        file_path = os.path.join(current_directory, file)
        print("Found file:", file_path)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Create an empty DataFrame for 'vx' increments
        df_increments = pd.DataFrame(columns=df.columns)

        # Calculate the increments for each 'vx' column
        for col in df.columns:
            if '_vx' in col:
                increments = df[col].diff().abs()  # Calculate the increments
                df_increments[col] = increments  # Assign the increments to the increments DataFrame

        # Plot the original 'vx' values
        plt.figure(figsize=(10, 6))
        for col in df.columns:
            if '_vx' in col:
                plt.scatter(df['timestamp'], df[col], label=col, s=2)
                plt.title("Original")

        # Keep only the 'z' columns in the original DataFrame
        df_z = df[[col for col in df.columns if '_z' in col]]

        # Combine df_increments with the 'z' columns
        df_combined = pd.concat([df_z, df_increments], axis=1)

        # Save the combined DataFrame to a new CSV file
        file_path_of = os.path.join(current_directory, "of_" + file)
        df_combined.to_csv(file_path_of, index=False)

        # Plot the 'vx' increments
        plt.figure(figsize=(10, 6))
        for col in df_increments.columns:
            plt.scatter(df['timestamp'], df_increments[col], label=col + ' Increment', s=2)

        # Plot settings
        plt.xlabel('Timestamp')
        plt.ylabel('vx Value')
        plt.title('vx Values and their Increments as a Function of Timestamp')
        plt.legend()
        plt.grid(True)

        # Show the plots
        plt.show()

# Function to shift a signal by a certain time offset
def shift_signal(signal, timestamps, offset):
    """
    Shift a signal by a certain time offset.

    Parameters:
    signal (np.array): The signal to be shifted.
    timestamps (np.array): The timestamps corresponding to the signal.
    offset (float): The time offset to shift the signal by.

    Returns:
    np.array: The shifted signal.
    """
    return np.interp(timestamps + offset, timestamps, signal)

def interpole_linear(common_timestamp, y1):
    """
    Perform linear interpolation on a signal to generate a finer resolution.

    Parameters:
    common_timestamp (np.array): The common timestamps for the signal.
    y1 (np.array): The signal values to be interpolated.

    Returns:
    np.array, np.array: The new timestamps and the interpolated signal.
    """
    # Create a linear interpolation function
    f_linear = interp1d(common_timestamp, y1, kind='linear')
    # Generate new x points for interpolation with a finer step
    common_timestamp = np.linspace(min(common_timestamp), max(common_timestamp), num=10000)  # 10000 points for higher continuity
    y1 = f_linear(common_timestamp)
    # Ensure that the new timestamps and interpolated signal have the same length
    assert len(common_timestamp) == len(y1)
    return common_timestamp, y1

@log_function_call
def synchro_data_v_v_e_z(file_raw_optics):
    """
    Synchronize and plot velocity and translation data from multiple CSV files.

    Parameters:
    file_raw_optics (str): Path to the raw optics file.

    Returns:
    tuple: Synchronized timestamps and smoothed velocity data.
    """
    # Plot of all three CSV files
    # plot_increment("data_robot_encoder/1b.csv", label='1b')
    # plot_increment("data_robot_encoder/2b.csv", label='2b')
    # plot_increment("data_robot_encoder/4b.csv", label='4b')

    # Plot settings
    # plt.title('Translation X increment normalized to meters per second')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Velocity [m/s]')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()

    # Read data from CSV files
    data_1b = pd.read_csv("data_robot_encoder/1b.csv", delimiter=",")
    data_2b = pd.read_csv("data_robot_encoder/2b.csv", delimiter=",")
    data_4b = pd.read_csv("data_robot_encoder/4b.csv", delimiter=",")

    # Extract translation X signals and timestamps
    translation_x_2b = data_2b['/tf/base/tool0_controller/translation/x']
    timestamps_2b = data_2b['__time']

    translation_x_1b = data_1b['/tf/base/tool0_controller/translation/x']
    timestamps_1b = data_1b['__time']

    translation_x_4b = data_4b['/tf/base/tool0_controller/translation/x']
    timestamps_4b = data_4b['__time']

    # Find the initial timestamp of the first signal
    start_time_1b = timestamps_1b[0]

    # Subtract the initial timestamp from the second signal's timestamps
    timestamps_2b_shifted = timestamps_2b - start_time_1b

    # Subtract the initial timestamp from the third signal's timestamps
    timestamps_4b_shifted = timestamps_4b - start_time_1b

    # Calculate the absolute difference between the three signals at each time step
    difference_signal = np.abs(translation_x_2b - translation_x_1b) + np.abs(translation_x_4b - translation_x_1b)

    FIND_SHIFTER = 0
    if FIND_SHIFTER:
        res_shift = []
        sh_1 = []
        sh_2 = []

        for shift_1 in np.arange(-2, 2, 0.1):
            for shift_2 in np.arange(-2, 2, 0.1):
                # Shift signals 2 and 4
                shifted_signal_2b = shift_signal(translation_x_2b, timestamps_2b, shift_1)
                shifted_signal_4b = shift_signal(translation_x_4b, timestamps_4b, shift_2)

                # Determine the minimum length between the two signals
                min_length = min(len(shifted_signal_2b), len(translation_x_1b), len(shifted_signal_4b))

                # Trim the end of the longer signal to match the shorter signal's length
                shifted_signal_2b = shifted_signal_2b[:min_length]
                shifted_signal_4b = shifted_signal_4b[:min_length]
                translation_x_1b = translation_x_1b[:min_length]

                # Calculate the absolute difference between the shifted signals at each time step
                difference_signal_shifted = np.abs(shifted_signal_2b - translation_x_1b) + np.abs(
                    shifted_signal_4b - translation_x_1b)

                # Calculate the mean difference over the entire time between the three signals
                mean_difference = np.mean(difference_signal_shifted)

                print(mean_difference)
                res_shift.append(mean_difference)
                sh_1.append(shift_1)
                sh_2.append(shift_2)

        serie_valori = [i + 1 for i in range(len(res_shift))]

        # Plot the difference between the shifted signals
        # plt.scatter(serie_valori, res_shift)
        # plt.scatter(serie_valori, sh_1)
        # plt.scatter(serie_valori, sh_2)
        # plt.xlabel('Time')
        # plt.ylabel('Absolute difference')
        # plt.title('Absolute difference between shifted signals')
        # plt.show()

        min_index = res_shift.index(min(res_shift))
        print("Shift 1 and 2", sh_1[min_index], sh_2[min_index])
        s11 = sh_1[min_index]
        s22 = sh_2[min_index]

    else:
        s11 = -0.99999999999
        s22 = -0.59999999999

    # Shift signals 2 and 4
    shifted_signal_2b = shift_signal(translation_x_2b, timestamps_2b, s11)
    shifted_signal_4b = shift_signal(translation_x_4b, timestamps_4b, s22)

    # Determine the minimum length between the two signals
    min_length = min(len(shifted_signal_2b), len(translation_x_1b), len(shifted_signal_4b))

    # Trim the end of the longer signal to match the shorter signal's length
    shifted_signal_2b = shifted_signal_2b[:min_length]
    shifted_signal_4b = shifted_signal_4b[:min_length]
    translation_x_1b = translation_x_1b[:min_length]

    # Find the minimum length between all signals
    min_length = min(len(translation_x_1b), len(shifted_signal_2b), len(shifted_signal_4b))

    # Calculate the original time step
    time_step = timestamps_1b.diff().mean()

    # Create a new timestamp based on the original time step
    common_timestamp = np.arange(0, min_length * time_step, time_step)

    # Plot the signals sharing the same x-axis
    y1 = translation_x_1b[:min_length]
    y2 = shifted_signal_2b[:min_length]
    y3 = shifted_signal_4b[:min_length]
    y1 = savgol_filter(y1, 3, 1)
    y2 = savgol_filter(y2, 3, 1)
    y3 = savgol_filter(y3, 3, 1)

    y1_series = pd.Series(y1)
    y2_series = pd.Series(y2)
    y3_series = pd.Series(y3)

    # Interpolate NaN values
    y1_interpolated = y1_series.interpolate(method='linear')
    y2_interpolated = y2_series.interpolate(method='linear')
    y3_interpolated = y3_series.interpolate(method='linear')

    # Calculate the mean of the three interpolated signals
    mean_position = np.nanmean(np.vstack([y1_interpolated, y2_interpolated, y3_interpolated]), axis=0)

    mean_velocity = np.diff(mean_position) / np.diff(common_timestamp)

    # Apply a 20-element moving average to the velocity signal
    window_size = 20
    mean_velocity_smoothed = np.convolve(mean_velocity, np.ones(window_size) / window_size, mode='valid')

    # Calculate the new length for time to match the moving average
    x_velocity = common_timestamp[1:]  # The timestamps for velocity are reduced by 1 compared to the original x
    x_smoothed = x_velocity[window_size - 1:]  #

    mean_velocity_smoothed_series = pd.Series(mean_velocity_smoothed)
    mean_velocity_smoothed_interpolated = mean_velocity_smoothed_series.interpolate(method='linear').to_numpy()

    plt.scatter(x_smoothed, mean_velocity_smoothed_interpolated, label='Smoothed Mean Velocity', s=3)
    # plt.scatter(common_timestamp, mean_position, label='Signal 1', s=1)
    # plt.scatter(common_timestamp, y2, label='Signal 1', s=1)
    # plt.scatter(common_timestamp, y3, label='Signal 1', s=1)

    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.title('Signals sharing the x-axis')
    plt.legend()
    plt.show()

    return x_smoothed, mean_velocity_smoothed_interpolated

def media_mobile(lista, window_size):
    """
    Calculate the moving average of a list with a specified window size.

    Parameters:
    lista (list): The list of values.
    window_size (int): The window size for calculating the moving average.

    Returns:
    list: The list of moving average values.
    """
    lista = np.array(lista)
    padding = window_size // 2  # Calculate the padding needed to maintain the same length as the input
    lista_padded = np.pad(lista, (padding, padding), mode='edge')  # Padding with the first/last value to maintain the same length
    moving_avg = np.convolve(lista_padded, np.ones(window_size) / window_size, mode='valid')
    return moving_avg[:len(lista)]  # Remove excess elements to maintain the same length as the input
def smoothing(x_fps, marker_n, window_size):
    """
    Function that splits a list into sublists based on marker ID changes and reassembles them while maintaining the original order.

    Parameters:
    x_fps (list): List of data to be split and reassembled.
    marker_n (list): List of marker IDs corresponding to the data.
    window_size (int): The window size for applying the moving average.

    Returns:
    list: Reassembled list with data in the original order.
    """
    data = x_fps
    sublists = []  # List to store sublists
    current_sublist = []  # Temporary list for the current sublist
    current_marker = None  # Current marker ID

    for i, (datum, marker) in enumerate(zip(data, marker_n)):
        # Check for marker change
        if current_marker != marker:
            if len(current_sublist) > window_size:
                current_sublist = media_mobile(current_sublist, window_size)
            sublists.append(current_sublist)
            current_sublist = []
            current_marker = marker

        # Add data to the current sublist
        current_sublist.append(datum)

    # Handle the last sublist (if present)
    if current_sublist:
        sublists.append(current_sublist)

    # Reassemble the data in the original order
    reassembled_data = []
    for sublist in sublists:
        reassembled_data.extend(sublist)

    return reassembled_data

def hist_adv(residui):
    """
    Plot an advanced histogram of residuals with a Gaussian distribution overlay.

    Parameters:
    residui (np.array): Array of residuals.

    """
    # Calculate systematic error
    errore_sistematico = np.mean(residui)

    # Calculate random error
    errore_casuale = np.std(residui)

    # Plot the histogram of residuals
    plt.hist(residui, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.6)

    # Calculate the standard deviation of the Gaussian distribution
    sigma_standard = np.std(residui)

    # Create an array of x values for the Gaussian distribution
    x_gauss = np.linspace(np.min(residui), np.max(residui), 100)

    # Calculate the y values corresponding to the Gaussian distribution
    y_gauss = norm.pdf(x_gauss, np.mean(residui), np.std(residui))

    # Plot the Gaussian distribution over the histogram of residuals
    plt.plot(x_gauss, y_gauss, 'r--', label='Gaussian Distribution')

    # Plot vertical lines corresponding to the standard deviation
    plt.axvline(x=errore_sistematico + errore_casuale, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=errore_sistematico - errore_casuale, color='k', linestyle='--', linewidth=1)
    # Add a vertical line corresponding to the mean value of the residuals
    plt.axvline(x=np.mean(residui), color='g', linestyle='-', linewidth=3)

    # Add the standard deviation to the title
    plt.title(f'Histogram of Residuals\nStandard Deviation: {sigma_standard:.4f}')

    plt.xlabel('Residuals [m]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_outlier(x, y):
    """
    Remove outliers from x and y based on the interquartile range.

    Parameters:
    x (pd.Series or list): Input data for x.
    y (pd.Series or list): Input data for y.

    Returns:
    tuple: Filtered x and y without outliers.
    """
    # Convert Pandas series to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate the first and third quartiles for x and y
    Q1_x, Q3_x = np.percentile(x, [10, 90])
    Q1_y, Q3_y = np.percentile(y, [10, 90])

    # Calculate the interquartile range for x and y
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y

    # Define the range to consider a value an outlier
    range_outlier = 1.5

    # Identify outliers in x
    outlier_x = (x < Q1_x - range_outlier * IQR_x) | (x > Q3_x + range_outlier * IQR_x)

    # Identify outliers in y
    outlier_y = (y < Q1_y - range_outlier * IQR_y) | (y > Q3_y + range_outlier * IQR_y)

    # Combine outliers found in both x and y
    outlier = outlier_x | outlier_y

    # Remove outliers from x and y
    x_filtrato = x[~outlier]
    y_filtrato = y[~outlier]

    # Print the number of outliers removed
    numero_outlier_rimossi = np.sum(outlier)
    print(f"Removed {numero_outlier_rimossi} outliers.")
    return x_filtrato, y_filtrato

def save_to_file_OF_results(filename, constant, constant_uncert, velocity):
    """
    Save optical flow results to a file.

    Parameters:
    filename (str): The name of the file to save the results.
    constant (float): The constant value to save.
    constant_uncert (float): The uncertainty of the constant value.
    velocity (float): The velocity value to save.
    """
    with open(filename, 'a') as file:
        file.write(f"{constant},{constant_uncert},{velocity}\n")
def weighted_linregress_with_error_on_y(x, y, y_err):
    """
    Perform weighted linear regression taking into account the uncertainty in y.

    Parameters:
    x (np.array): Independent variable data.
    y (np.array): Dependent variable data.
    y_err (np.array): Uncertainty in y.

    Returns:
    tuple: Slope, intercept, and R^2 of the regression line.
    """
    # Weights based on the uncertainty in y
    w = 1 / y_err

    # Calculate the weighted means of x and y
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)

    # Calculate the weighted covariances
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    cov_xx = np.sum(w * (x - x_mean) ** 2)

    # Calculate the weighted regression coefficient and intercept
    slope = cov_xy / cov_xx
    intercept = y_mean - slope * x_mean

    # Calculate R^2 considering only the uncertainty in y
    residui = y - (slope * x + intercept)
    somma_quadri_residui = np.sum(w * residui ** 2)
    totale = np.sum(w * (y - y_mean) ** 2)
    r_squared = 1 - (somma_quadri_residui / totale)

    return slope, intercept, r_squared

def smart_cutter_df(df, threshold):
    """
    Split a DataFrame into sub-DataFrames based on a threshold for frame discontinuities.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing frame data.
    threshold (int): Threshold for identifying frame discontinuities.

    Returns:
    list: List of sub-DataFrames.
    """
    start_idx = 0
    sub_dataframes = []
    for i in range(1, len(df)):
        if df['n_frame'].iloc[i] - df['n_frame'].iloc[i - 1] > threshold:
            # If there is a discontinuity
            sub_dataframes.append(df.iloc[start_idx:i])
            start_idx = i
    sub_dataframes.append(df.iloc[start_idx:])
    return sub_dataframes

def delete_static_data_manually(df, marker_riferimento, confidence_delation):
    """
    Manually delete static data based on the reference marker position and confidence deletion factor.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing marker data.
    marker_riferimento (str): Reference marker column name.
    confidence_delation (float): Confidence deletion factor for determining thresholds.

    Returns:
    pd.DataFrame: Filtered DataFrame with static data removed.
    """
    # Calculate the minimum and maximum values of the reference marker position
    x_min = df[marker_riferimento].min()
    x_max = df[marker_riferimento].max()

    # Calculate the range of x values considering the confidence deletion factor
    x_range = x_max - x_min
    x_range *= (1 - confidence_delation)

    # Calculate the threshold values
    x_threshold_min = x_min + confidence_delation * x_range
    x_threshold_max = x_max - confidence_delation * x_range

    # Filter rows that do not meet the threshold criteria
    df_filtered = df[(df[marker_riferimento] >= x_threshold_min) & (df[marker_riferimento] <= x_threshold_max)]

    return df_filtered

def imaga_analizer_raw():
    """
    Analyze raw image data from MP4 files, extract marker positions, and save the results to a CSV file.
    """
    # Starting directory
    start_directory = os.getcwd()

    # Raw acquisition directory
    acquisition_raw_directory = os.path.join(start_directory, 'aquisition_raw')

    # Iterate over all MP4 files in the raw acquisition directory and its subdirectories
    for mp4_file in iter_mp4_files(acquisition_raw_directory):
        print("MP4 file found:", mp4_file)

        cap = cv2.VideoCapture(mp4_file)

        PROC = 1
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize the processed frame counter
        processed_frames = 0

        n_frames = 0

        if PROC:
            # Delete the file if it already exists

            header = ['timestamp', '7_z', '7_vx', '8_z', '8_vx', '9_z', '9_vx', '10_z', '10_vx', '11_z', '11_vx']
            df = pd.DataFrame(columns=header)

        while cap.isOpened():
            n_frames += 1
            row_data = {'n_frame': n_frames}

            ret, frame = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            row_data = {'timestamp': timestamp}
            if not ret:
                break

            processed_frames += 1

            # Print progress at regular intervals
            if processed_frames % 100 == 0:  # Print every 100 frames
                print(f"Processed frames: {processed_frames}/{total_frames}")

            if PROC:
                # Calculate the height of the image
                height = frame.shape[0]

                # Calculate the new height after cropping
                new_height = int(height * 0.3)  # Remove one third of the height

                # Crop the image
                frame = frame[new_height:, :]

            # Find ArUco markers in the frame
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Example of contrast and brightness adjustment
            alpha = 2  # Contrast factor
            beta = 5  # Brightness factor
            gray_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

            if PROC:
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

                if ids is not None:
                    # Draw detected markers on the frame
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Loop through the found markers
                    for i in range(len(ids)):
                        # Calculate the 3D position of the marker relative to the camera
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, mtx, dist)
                        marker_position = tvec[0][0]  # 3D position of the marker
                        x, y, z = marker_position[0], marker_position[1], marker_position[2]

                        # Extract the marker ID
                        marker_id = ids[i][0]

                        # Calculate the x-coordinates of the marker corners
                        x_coords = corners[i][0][:, 0]

                        # Calculate the approximate x-coordinate of the marker center
                        center_x = np.mean(x_coords)
                        z_key = f'{marker_id}_z'
                        vx_key = f'{marker_id}_vx'

                        row_data[z_key] = z
                        row_data[vx_key] = center_x

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if PROC:
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # Release resources and save the DataFrame to a CSV file
        if PROC:
            # Save the DataFrame to a CSV file
            current_directory = os.getcwd()

            # Count how many CSV files start with "raw_re_output" in the current directory
            count = sum(
                1 for file in os.listdir(current_directory) if file.startswith("raw_re_output") and file.endswith(".csv"))
            df.to_csv('raw_re_output_' + str(count + 1) + '.csv', index=False)

        cap.release()
        cv2.destroyAllWindows()

def plotter_raw(df):
    """
    Plot the raw coordinates of markers from the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the marker coordinates.

    Returns:
    None: Displays scatter plots of the marker coordinates.
    """
    # Iterate through marker IDs from 1 to 5
    for marker_inr in marker_ids:
        print("marker:", marker_inr)
        # Select rows with non-null values for the current marker
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]

        # Plot the x and z coordinates of the current marker as scatter plots
        plt.scatter(marker_data['n_frame'], marker_data[f'x_ip_{marker_inr}'] / 2000, label=f'X Coordinate Marker {marker_inr}', color='blue')
        plt.scatter(marker_data['n_frame'], marker_data[f'z_3D_{marker_inr}'], label=f'Z Coordinate Marker {marker_inr}', color='red')
        plt.scatter(marker_data['n_frame'], marker_data[f'x_3D_{marker_inr}'], label=f'X_3D Coordinate Marker {marker_inr}', color='green')

        # Add labels and legend to the current plot
        plt.xlabel('Frame Number')
        plt.ylabel('Coordinates')
        plt.title(f'X, X_3D and Z Coordinates of Marker {marker_inr}')
        plt.legend()

        # Show the current plot
        plt.show()


def save_results_to_excel(results, output_excel):
    """
    Save results to an Excel file.

    Parameters:
    results (list of dict): List of dictionaries containing the results.
    output_excel (str): Path to the output Excel file.

    Returns:
    None: Saves the results to the specified Excel file.
    """
    # Read the existing Excel file if it exists
    if os.path.exists(output_excel):
        df = pd.read_excel(output_excel)
    else:
        df = pd.DataFrame()  # Create a new DataFrame if the Excel file does not exist

    # Iterate over the results and add each dictionary item as a row in the DataFrame
    for result in results:
        dict_to_add = {}
        # Extract the value for each column from the dictionary and add it as a row to the DataFrame
        for key, value_list in result.items():
            # Check if the key (header) already exists in the DataFrame
            if key in df.columns:
                # If the column already exists, extend the existing Series with the new values
                dict_to_add[key] = value_list
        df = df.append(pd.DataFrame(dict_to_add))

    # Save the updated DataFrame to the Excel file
    df.to_excel(output_excel, index=False)

# #
# Main script for calling key functions in the problem

# Flag for processing raw video to extract information
RAW_VIDEO_PROCESSING = 0

if RAW_VIDEO_PROCESSING:
    # Function to analyze raw video and extract necessary data
    imaga_analizer_raw()

    # Function to convert positional data from the video into optical flow data
    convert_position_to_speed()

# Uncomment the following line if intermediate conversion is needed independently
# convert_position_to_speed()

# Flag for performing experimental model fitting and graph generation
EXPERIMENTAL_MODEL_FITTING = 0

if EXPERIMENTAL_MODEL_FITTING:
    # Path to the data file for experimental model fitting
    file_path_1 = 'dati_of/all_points_big_fix_speed.xlsx'

    # Function to show results from the experimental file
    show_result_ex_file(file_path_1)

    # Function to perform windowing analysis and calculate uncertainties
    windowing_vs_uncertanty(file_path_1)

# Flag for evaluating experimental K constants from the model at different reference speeds
EXP_Ks_RESULTS_EVALUATION = 0

if EXP_Ks_RESULTS_EVALUATION:
    # Function to analyze the constant K extracted from the model evaluation at different speeds
    constant_analisis()

# Flag for synchronizing robot velocities and validating raw optical flow data
RAW_ROBOT_AND_RAW_OPTICAL_FLOW_VALIDATION = 1

if RAW_ROBOT_AND_RAW_OPTICAL_FLOW_VALIDATION:
    # Function to synchronize external velocities of the robot to ensure a reliable path
    x_s, vy_s = synchro_data_v_v_e_z("results_raw.xlsx")

    # Function to merge robot raw data with optical flow and depth information, synchronizing and validating the dataset
    merge_dataset_extr_int(x_s, vy_s)

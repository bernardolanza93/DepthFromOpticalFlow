import sys
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import norm
from itertools import groupby
from statistics import mean
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.signal import decimate
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), 'utility_library'))
from multi_marker_source_function import *
import matplotlib.font_manager

from decorator import *




@log_function_call
def find_signal_boundaries(signal, threshold=0.1):
    """
    Trova gli istanti di inizio e fine di un segnale che subisce variazioni repentine.

    Parameters:
    signal (np.array): Il segnale di input.
    threshold (float): La soglia per identificare variazioni significative nel segnale.

    Returns:
    (int, int): Istanti di inizio e fine del segnale significativo.
    """
    # Calcola la derivata del segnale
    derivative = np.diff(signal)

    # Identifica gli indici dove la variazione è significativa

    significant_changes = np.where(np.abs(derivative) > threshold)[0]

    if len(significant_changes) == 0:
        return None, None

    # Prendi il primo e l'ultimo cambiamento significativo
    start_idx = significant_changes[0]
    end_idx = significant_changes[-1]

    # Trova l'inizio reale prima del primo cambiamento significativo
    for i in range(start_idx, 0, -1):
        if np.abs(signal[i]) < threshold:
            start_idx = i
            break

    # Trova la fine reale dopo l'ultimo cambiamento significativo
    for i in range(end_idx, len(signal) - 1):
        if np.abs(signal[i]) < threshold:
            end_idx = i
            break

    return start_idx, end_idx

@log_function_call
def calculate_theo_model_and_analyze(n_df,n_vy,win, vlim):
    # Separa le colonne in base al suffisso '_vx' e '_z'
    vx_columns = [col for col in n_df.columns if '_vx' in col]
    z_columns = [col for col in n_df.columns if '_z' in col]

    #result = [x * constant for x in n_vy]
    # Vettori per raccogliere i dati
    all_z = []
    all_vx = []
    all_distance = []
    all_vy_robot = []


    for vx_col, z_col in zip(vx_columns, z_columns):
        # Calcola la lista dist
        smoothed_vx = n_df[vx_col].rolling(window=win).mean()
        n_df[vx_col] = smoothed_vx

        dist = []
        for i in range(len(n_df)):
            if n_vy[i] >  vlim:
                if n_df[vx_col].iloc[i] != 0:
                    if n_df[vx_col].iloc[i] != 0:

                        dist.append(n_vy[i] * fx / (n_df[vx_col].iloc[i] * 60))
                    else:
                        dist.append(np.nan)  # Oppure qualsiasi altro valore per gestire la divisione per zero

                    # Aggiungi i valori ai vettori complessivi
                    all_z.append(n_df[z_col].iloc[i])
                    all_vx.append((n_df[vx_col].iloc[i] * 60))
                    all_distance.append(n_vy[i] * fx / (n_df[vx_col].iloc[i] * 60))
                    all_vy_robot.append(n_vy[i])



    # Converti le liste in array numpy
    all_z = np.array(all_z)
    all_distance = np.array(all_distance)

    # Rimuovi i valori NaN
    mask = ~np.isnan(all_z) & ~np.isnan(all_distance)
    all_z = all_z[mask]
    all_distance = all_distance[mask]

    # Fitta il modello Y = X
    y_pred = all_z  # Modello Y = X prevede che y_pred è uguale a x

    # Calcola R^2
    r2 = r2_score(all_distance, y_pred)

    # Calcola altre metriche di bontà del fit
    mse = mean_squared_error(all_distance, y_pred)
    mae = mean_absolute_error(all_distance, y_pred)
    rmse = np.sqrt(mse)

    # Calcola la dispersione media
    dispersione_media = np.mean(np.abs(all_distance - y_pred))

    print("R^2:", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Dispersione Media:", dispersione_media)

    return  all_z , all_distance

@log_function_call
def merge_dataset_extr_int(x, vy ):

    vy = abs(vy)
    # # Plotta il segnale di velocità medio e il segnale di velocità medio filtrato



    # Percorso del file CSV
    file_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/of_raw_re_output_1.csv'


    # Carica il file CSV
    df = pd.read_csv(file_path)

    # Rimuovi le colonne che contengono "164"
    df = df.loc[:, ~df.columns.str.contains('164')]

    # Rimuovi le colonne con zero valori non nulli
    df = df.dropna(axis=1, how='all')

    # Filtra i dati fino al 3300-esimo record
    df = df.loc[:3299]

    # Poiché la colonna timestamp sembra non avere dati validi, dobbiamo generare un timestamp
    df['timestamp'] = range(len(df))
    t = df['timestamp']

    # Separa le colonne in base al suffisso '_z' e '_vx'

    x_vy  =x

    FIND_SHIFTER  =0

    if FIND_SHIFTER:
        mean_diff_s_robot = []
        mean_diff_s_cam = []
        meanss = []
        for shift_rob in np.arange(0,0.006,0.0001):
            for shift_cam in np.arange(0, 4, 0.1):


                ini, endi = find_signal_boundaries(df["11_vx"],shift_cam)
                ini_rob, endi_rob = find_signal_boundaries(vy, shift_rob)

                if ini is not None and endi is not None:
                    # Taglia tutti i segnali del DataFrame utilizzando questi indici
                    n_df = df.iloc[ini - 2:endi + 3].reset_index(drop=True)

                if ini_rob is not None and endi_rob is not None:
                    n_vy = vy[ini_rob:endi_rob + 1]
                    n_x_vy = x_vy[ini_rob:endi_rob + 1]

                n_df['timestamp'] = n_df['timestamp'] - n_df['timestamp'].iloc[0]
                n_x_vy = n_x_vy - n_x_vy[0]
                # Metodo 1: Utilizzo di slicing
                factor = len(n_vy) // len(n_df["11_vx"])
                # Crea un nuovo asse temporale per il segnale decimato
                x_vy_decimated = np.linspace(n_df["timestamp"].iloc[0], n_df["timestamp"].iloc[-1], len(n_df["11_vx"]))

                # Interpola il segnale n_vy sul nuovo asse temporale
                interpolator = interp1d(np.linspace(0, len(n_vy) - 1, len(n_vy)), n_vy, kind='linear')
                n_vy_decimated = interpolator(np.linspace(0, len(n_vy) - 1, len(n_df["11_vx"])))
                n_vy = n_vy_decimated
                n_x_vy = x_vy_decimated
                _11_esay = (n_df["11_vx"]).tolist()
                massimo_segnale2 = max(_11_esay)
                #print(massimo_segnale2)
                max1 = max(n_vy)
                #segnale_normalizzato = [element * massimo_segnale2 / max1 for element in n_vy]
                segnale_normalizzato = n_vy / max1 * massimo_segnale2

                pointwise_difference = [abs(a - b) for a, b in zip(segnale_normalizzato, _11_esay)]
                # Converti la lista in un array numpy
                data_array = np.array(pointwise_difference)

                # Calcola la media ignorando i NaN
                mean_difference = np.nanmean(data_array)



                meanss.append(mean_difference)
                mean_diff_s_robot.append(shift_rob)
                mean_diff_s_cam.append(shift_cam)
                print(mean_difference)

                # Metodo 2: Utilizzo di SciPy decimate



        range_list = np.linspace(0, len(meanss) - 1, len(meanss), dtype=int)  # Ensure integers




        plt.scatter(range_list,meanss)
        plt.show()
        # Converti la lista in un array numpy
        data_array = np.array(meanss)

        # Trova l'indice del valore minimo ignorando i NaN
        min_index = np.nanargmin(data_array) # Get the index of the maximum value in y
        thres_robot = mean_diff_s_robot[min_index]
        thres_cam = mean_diff_s_cam[min_index]
        print(min_index,thres_robot,thres_cam)
    else:
        thres_robot =  0.0035
        thres_cam = 1.0


    ini, endi = find_signal_boundaries(df["11_vx"], thres_cam)
    ini_rob, endi_rob = find_signal_boundaries(vy, thres_robot)

    if ini is not None and endi is not None:
        # Taglia tutti i segnali del DataFrame utilizzando questi indici
        n_df = df.iloc[ini - 2:endi + 3].reset_index(drop=True)

    if ini_rob is not None and endi_rob is not None:
        n_vy = vy[ini_rob:endi_rob + 1]
        n_x_vy = x_vy[ini_rob:endi_rob + 1]

    n_df['timestamp'] = n_df['timestamp'] - n_df['timestamp'].iloc[0]
    n_x_vy = n_x_vy - n_x_vy[0]
    # Metodo 1: Utilizzo di slicing
    factor = len(n_vy) // len(n_df["11_vx"])
    # Crea un nuovo asse temporale per il segnale decimato
    x_vy_decimated = np.linspace(n_df["timestamp"].iloc[0], n_df["timestamp"].iloc[-1], len(n_df["11_vx"]))

    # Interpola il segnale n_vy sul nuovo asse temporale
    interpolator = interp1d(np.linspace(0, len(n_vy) - 1, len(n_vy)), n_vy, kind='linear')
    n_vy_decimated = interpolator(np.linspace(0, len(n_vy) - 1, len(n_df["11_vx"])))
    n_vy = n_vy_decimated
    n_x_vy = x_vy_decimated
    _11_esay = (n_df["11_vx"]).tolist()
    massimo_segnale2 = max(_11_esay)
    # print(massimo_segnale2)
    max1 = max(n_vy)
    # segnale_normalizzato = [element * massimo_segnale2 / max1 for element in n_vy]
    segnale_normalizzato = n_vy / max1 * massimo_segnale2










    n_vx_columns = [col for col in n_df.columns if '_vx' in col]
    n_z_columns = [col for col in n_df.columns if '_z' in col]

    # Crea un plot con tre subplot
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Primo subplot per vx_columns
    for col in n_vx_columns:
        axs[0].plot(n_df['timestamp'], n_df[col], label=col)
    axs[0].plot(n_df['timestamp'], segnale_normalizzato, label="ROB")
    axs[0].set_xlabel('Timestamp')
    axs[0].set_ylabel('VX Values')
    axs[0].set_title('Original VX Signals')
    axs[0].legend()
    axs[0].grid(True)

    # Secondo subplot per vy
    axs[1].plot(n_x_vy, n_vy, label='ROBOT', linestyle='--', color='black')
    axs[1].set_xlabel('Tempo')
    axs[1].set_ylabel('Velocità (vy)')
    axs[1].set_title('Ground Truth (vy)')
    axs[1].legend()
    axs[1].grid(True)

    # Terzo subplot per z_columns
    for col in n_z_columns:
        axs[2].plot(n_df['timestamp'], n_df[col], label=col)
    axs[2].set_xlabel('Timestamp')
    axs[2].set_ylabel('Z Values')
    axs[2].set_title('Original Z Signals')
    axs[2].legend()
    axs[2].grid(True)

    # Mostra i plot
    plt.tight_layout()
    plt.show()



    all_z, all_distance  = calculate_theo_model_and_analyze(n_df, n_vy, 1 , 0.2)
    all_z_3, all_distance_3 = calculate_theo_model_and_analyze(n_df, n_vy, 5, 0.23)



    # Plot delle relazioni
    plt.figure(figsize=(10, 5))
    plt.scatter(all_z, all_distance, label='Distance vs VX', alpha=0.5, s = 1)
    plt.scatter(all_z_3, all_distance_3, label='Distance vs VX media 3', alpha=0.5, s = 1)
    min_val = min(min(all_z), min(all_distance))
    max_val = max(max(all_z), max(all_distance))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfetta previsione (y = x)')



    #plt.scatter(all_vx, all_z, label='Z vs VX', alpha=0.5)
    plt.xlabel('reali')
    plt.ylabel('predetti')
    plt.title('Relazione tra VX, Distance e Z')
    plt.legend()

    plt.grid(True)
    plt.show()



def modello(x, costante):
    return costante / x

def compute_dz(Vx, Vx_prime, fx, fy, cx, cy):
    dz = ((Vx * fx) / Vx_prime )

    return dz


@log_function_call
def windowing_vs_uncertanty(file_path):
    SHOW_PLOT = 0

    v_ext = []
    unc_k = []
    sigma_gauss = []
    win_size = []

    for window_size in range(1,10,1):


        # Elimina il file constant se esiste
        if os.path.exists("constant.txt"):
            os.remove("constant.txt")

        # Crea il file constant con gli header
        with open("constant.txt", 'w') as file:
            file.write("constant,constant_uncert,velocity\n")


        data = pd.read_excel(file_path)
        # Rendi positivi i valori di vx e vx_std

        # Rendi positivi i valori di vx e vx_std
        data['vx'] = abs(data['vx'])


        # Rimuovi le righe con zeri o valori mancanti nella riga
        data = data[(data != 0).all(1)]

        # Dividi il DataFrame in base al valore della colonna vx_3D
        gruppi = data.groupby('vx_3D')

        # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
        sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}

        for chiave, valore in sotto_dataframe.items():
            print(chiave, valore)
            data = sotto_dataframe[chiave]



            # Definisci i colori per i diversi valori di vx_3D
            color_map = {
                v1: 'red',
                v2: 'azure',
                v3: 'green',
                v4: 'orange',
                v5: 'purple'
            }
            print(fx,fy,cx,cy)



            x_fps = data['vx']





            marker_n = data['marker']
            x = [element * 60 for element in x_fps]


            y = data['z_mean']

            SMOOTHING = 1
            window = 0


            if SMOOTHING:
                window = 7
                x_or = x
                x_s = smoothing(x,marker_n, window_size)
                x_s_graph = [x_ii + 1000 for x_ii in x_s]
                x = x_s

            #x = media_mobile(x,150)


            color_p = color_map[chiave]

            PLOT_OF_RAW  = 1
            if PLOT_OF_RAW and SMOOTHING:


                x__1 = list(range(len(x)))
                #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
                plt.plot(x__1, x_or)
                plt.plot(x__1, x_s_graph)
                marker_aug =  [element * 100 for element in marker_n]
                plt.plot(x__1,marker_aug)
                if SHOW_PLOT:
                    plt.show()



            # Vx_prime_values = np.linspace(min(x), max(x), 100)
            Vx_prime_values = sorted(x)


            # Adatta il modello ai dati
            parametri, covarianza = curve_fit(modello, x, y)


            # Estrai la costante stimata
            costante_stimata = parametri[0]

            # Calcola l'incertezza associata alla costante
            incertezza_costante = np.sqrt(np.diag(covarianza))[0]

            # Calcola l'R^2
            residui = y - modello(x, costante_stimata)
            somma_quadri_residui = np.sum(residui ** 2)
            totale = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (somma_quadri_residui / totale)

            # Calcola i valori del modello per il plotting
            x_modello = np.linspace(min(x), max(x), 100)
            y_modello = modello(x_modello, costante_stimata)

            # Salva i dati nel file
            save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)

            plt.figure(figsize=(15, 10))

            # Grafico dei punti grezzi e del modello
            plt.scatter(x, y, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")



            #MODELLO GENERICO


            plt.plot(x_modello, y_modello, label='Modello genereico Dz = k/OF', color='black',linestyle='-.',)

            plt.xlabel('OF [px/s]')
            plt.ylabel('Depth [m]')
            plt.grid(True)
            plt.ylim(0, 2.1)

            # Plot aggiunto
            Y_teorico = []
            for i in range(len(Vx_prime_values)):


                dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
                Y_teorico.append(dzi)

            plt.plot(Vx_prime_values, Y_teorico, color="grey",label='Modello teorico Dz = (V_r * fx)/OF')


            # Calcola l'errore sistematico
            residui = (y - Y_teorico) / y
            errore_sistematico = np.mean(residui)

            # Calcola l'errore casuale
            errore_casuale = np.std(residui)
            # Calcola i residui

            costante_teorica = fx * float(chiave)

            plt.title(
                f'depth vs Optical flow [z = k / vx] - media mobile filtro :{window}, \n K_th: {costante_teorica:.2f} , K_exp:{costante_stimata:.2f} +- {incertezza_costante:.2f} [px*m]  o [px * m/s] || R^2:{r_squared:.4f} \n Stat on relative residuals (asimptotic - no gaussian): \n epsilon_sistem_REL :  {errore_sistematico*100 :.3f}% , sigma_REL: {errore_casuale*100 :.3f} %')


            # Posiziona la legenda in alto a destra
            plt.legend(loc="upper right")






            if SHOW_PLOT:
                plt.show()

            if SHOW_PLOT:

                hist_adv(residui)

            v_ext.append(color_p)
            unc_k.append(incertezza_costante)
            sigma_gauss.append(errore_casuale)
            win_size.append(window_size)


    plt.close('all')
    # Creazione dei subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Grafico 1: Incertezza associata ai parametri del modello
    for i in range(len(v_ext)):
        ax1.scatter(win_size[i], unc_k[i], color=v_ext[i], marker="x", label='Model ' + str(i + 1))
    ax1.set_xlabel('Window Size [samples]')
    ax1.set_ylabel('k uncertanty [m*px]')


    # Grafico 2: Sigma del modello fittato (Sigma Gauss)
    for i in range(len(v_ext)):
        ax2.scatter(win_size[i], sigma_gauss[i], color=v_ext[i], label='Model ' + str(i + 1))
    ax2.set_xlabel('Window Size [samples]')
    ax2.set_ylabel('relative sigma of residuals [std]')

    # Imposta il titolo del subplot
    fig.suptitle('Model Evaluation - moving avarege effect')

    # Mostra il plot
    plt.show()

# Funzione per calcolare la distanza euclidea

def calculate_distance_vector(x1, y1, x2_array, y2_array):
    return np.sqrt((x1 - x2_array) ** 2 + (y1 - y2_array) ** 2)

@log_function_call
def show_result_ex_file(file_path):
    SHOW_PLOT = 1


    # Elimina il file constant se esiste
    if os.path.exists("constant.txt"):
        os.remove("constant.txt")

    # Crea il file constant con gli header
    with open("constant.txt", 'w') as file:
        file.write("constant,constant_uncert,velocity\n")


    data = pd.read_excel(file_path)
    # Rendi positivi i valori di vx e vx_std

    # Rendi positivi i valori di vx e vx_std
    data['vx'] = abs(data['vx'])


    # Rimuovi le righe con zeri o valori mancanti nella riga
    data = data[(data != 0).all(1)]

    # Dividi il DataFrame in base al valore della colonna vx_3D
    gruppi = data.groupby('vx_3D')

    # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
    sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}

    for chiave, valore in sotto_dataframe.items():
        print(chiave, valore)
        data = sotto_dataframe[chiave]



        # Definisci i colori per i diversi valori di vx_3D
        color_map = {
            v1: 'red',
            v2: 'cyan',
            v3: 'green',
            v4: 'orange',
            v5: 'purple'
        }
        print(fx,fy,cx,cy)



        x_fps = data['vx']





        marker_n = data['marker']
        x = [element * 60 for element in x_fps]


        y = data['z_mean']

        SMOOTHING = 0
        window = 0


        if SMOOTHING:
            window = 3
            x_or = x
            x_s = smoothing(x,marker_n, window)
            x_s_graph = [x_ii + 1000 for x_ii in x_s]
            x = x_s

        #x = media_mobile(x,150)


        color_p = color_map[chiave]

        PLOT_OF_RAW  = 1
        if PLOT_OF_RAW and SMOOTHING:
            x__1 = list(range(len(x)))
            #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
            plt.plot(x__1, x_or)
            plt.plot(x__1, x_s_graph)
            marker_aug =  [element * 100 for element in marker_n]
            plt.plot(x__1,marker_aug)
            if SHOW_PLOT:
                plt.show()



        # Vx_prime_values = np.linspace(min(x), max(x), 100)
        Vx_prime_values = sorted(x)


        # Adatta il modello ai dati
        parametri, covarianza = curve_fit(modello, x, y)


        # Estrai la costante stimata
        costante_stimata = parametri[0]

        # Calcola l'incertezza associata alla costante
        incertezza_costante = np.sqrt(np.diag(covarianza))[0]

        # Calcola l'R^2
        residui = y - modello(x, costante_stimata)
        somma_quadri_residui = np.sum(residui ** 2)
        totale = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (somma_quadri_residui / totale)

        # Calcola i valori del modello per il plotting
        x_modello = np.linspace(min(x), max(x), len(x))
        y_modello = modello(x_modello, costante_stimata)

        # Salva i dati nel file
        save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)

        # Creazione del grafico
        plt.figure(figsize=(12, 6))

        # Plot aggiunto
        Y_teorico = []
        for i in range(len(Vx_prime_values)):
            dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
            Y_teorico.append(dzi)

        # Calcolo della distanza minima per ogni punto sperimentale
        min_distances = []
        # Creazione del DataFrame
        data = pd.DataFrame({
            'x': x,
            'y': y
        })
        modello_q = pd.DataFrame({
            'x_modello': x_modello,
            'y_modello': y_modello
        })

        for index, row in data.iterrows():
            distances = calculate_distance_vector(row['x'], row['y'], modello_q['x_modello'].values,
                                                  modello_q['y_modello'].values)
            min_distance = np.min(distances)
            min_distances.append(min_distance)

        data['min_distance'] = min_distances

        # Visualizzazione dei risultati
        print(data[['x', 'y', 'min_distance']])

        # Calcolo dell'errore medio assoluto (MAE)
        mae = np.mean(min_distances)
        print(f'Errore medio assoluto: {mae:.4f}')


        # Preparazione dei dati
        data = pd.DataFrame({
            'x': x,
            'y': y,
            'x_modello': x_modello,
            'y_modello': y_modello,
            'Vx_prime_values': Vx_prime_values,
            'Y_teorico' : Y_teorico
        })

        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'Nimbus Sans',  # Puoi cambiare 'DejaVu Sans' con il font desiderato
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.titleweight': 'bold',
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        # Grafico dei punti grezzi e del modello
        sns.scatterplot(x='x', y='y', data=data, label='Dati grezzi', color=color_p, s=50, alpha=0.7, marker="^",
                        edgecolor="black")

        sns.lineplot(x='x_modello', y='y_modello', data=data, label=r'Experimental model $d = k_{{exp}}/v_{{px}}$',
                     color='black', linestyle='-.')
        sns.lineplot(x='Vx_prime_values', y='Y_teorico', data=data, color="grey",
                     label=r'Analytical model $d = V_{{ext}} ⋅ f_{{y}}/v_{{px}}$', alpha=0.7, linewidth=2)

        plt.xlabel(r'$v_{px}$ [$px/s$]', fontsize=16)
        plt.ylabel('Depth [m]', fontsize=16)
        plt.ylim(0, 2.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Calcola l'errore sistematico
        residui = (data['y'] - data['Y_teorico']) / data['y']
        errore_sistematico = np.mean(residui)

        # Calcola l'errore casuale
        errore_casuale = np.std(residui)

        costante_teorica = fx * float(chiave)

        # Titolo del grafico
        plt.title(
            f' Optical pixel displacement vs. depth. Performed at $V_{{ext}}$ = {chiave} m/s ',
            fontsize=18, fontweight='bold', pad=15)

        # Posiziona la legenda in alto a destra
        plt.legend(loc="upper right", fontsize=14)

        # Percorso del file di salvataggio
        file_path_fig = 'results/speed_' + str(chiave) + '_k_model.png'

        # Verifica se il file esiste già
        if os.path.exists(file_path_fig):
            # Se il file esiste, eliminilo
            os.remove(file_path_fig)
            print("removed old plot")

        # Salva la figura
        plt.savefig(file_path_fig, dpi=300, bbox_inches='tight')


        if SHOW_PLOT:
            plt.show()

@log_function_call
def constant_analisis():
    # Leggi i dati dal file
    data = np.loadtxt("constant.txt", delimiter=',', skiprows=1)

    # Estrai le colonne
    constant_data = data[:, 0]
    constant_uncert_data = data[:, 1]
    velocity_data = data[:, 2]

    # Fai la regressione lineare tenendo conto dell'incertezza sulla costante
    slope, intercept, r_squared = weighted_linregress_with_error_on_y(velocity_data, constant_data, 1 / constant_uncert_data)
    # Calcola l'incertezza della pendenza
    residuals = constant_data - (slope * velocity_data + intercept)
    uncert_slope = np.sqrt(np.sum(constant_uncert_data ** 2 * residuals ** 2) / np.sum((velocity_data - np.mean(velocity_data)) ** 2))
    # Calcola l'R^2

    sigma3 = [element * 3 for element in constant_uncert_data]

    plt.figure(figsize=(12, 7))
    # Grafico
    plt.scatter(velocity_data, constant_data, label='Dati',s = 15)
    plt.errorbar(velocity_data, constant_data, yerr=sigma3, fmt='none', label='Incertezza')
    plt.plot(velocity_data, slope * velocity_data + intercept, color='red', label='k(v_ext) sperimentale')
    plt.plot(velocity_data, velocity_data* fx, color='orange', label='K(v_ext) teorico')
    plt.xlabel('V_ext[m/s]')
    plt.ylabel('Constant [k]')
    plt.title(
        f' k_i = f(v_ext) : slope:{slope:.1f} sigma:{uncert_slope:.1f} k/[m/s]|| R^2:{r_squared:.4f} \n incertezza su parametri: {constant_uncert_data[0]:.2f} , {constant_uncert_data[1]:.2f},{constant_uncert_data[2]:.2f},{constant_uncert_data[3]:.2f} [px*m] - 99.7% int')
    plt.legend()
    plt.grid(True)



    # Percorso del file di salvataggio
    file_path_fig = 'results/k_LR.png'

    # Verifica se il file esiste già
    if os.path.exists(file_path_fig):
        # Se il file esiste, eliminilo
        os.remove(file_path_fig)
        print("removed old plot")

    # Salva la figura
    plt.savefig(file_path_fig)
    plt.show()



def interpolate_signal(signal_ref, timestamps_ref, signal_other, timestamps_other):
    # Interpolazione del segnale
    interpolated_signal_other = np.interp(timestamps_ref, timestamps_other, signal_other)
    return interpolated_signal_other
def plot_increment(file_path, label):
    # Leggi i dati dal file CSV
    data = pd.read_csv(file_path, delimiter=",")

    # Estrai il timestamp minimo
    min_timestamp = data['__time'].min()

    # Calcola i timestamp relativi a zero
    timestamps = data['__time'] - min_timestamp

    # Estrai i dati relativi alla traslazione X
    translation_x = data['/tf/base/tool0_controller/translation/x']

    # Calcola l'incremento della traslazione X
    translation_increment = translation_x.diff()

    # Calcola l'intervallo temporale tra i punti
    time_diff = timestamps.diff()

    # Calcola la velocità in metri al secondo
    velocity_mps = translation_increment / time_diff

    # Plot dell'incremento di traslazione X normalizzato in metri al secondo
    plt.plot(timestamps[1:], velocity_mps[1:], label=label)

def iter_mp4_files(directory):
    # Itera su tutti i file e le directory nella directory specificata
    for root, dirs, files in os.walk(directory):
        # Itera su tutti i file
        for file in files:
            # Verifica se il file ha estensione MP4
            if file.endswith('.MP4'):
                # Restituisce il percorso completo del file MP4
                print(os.path.join(root, file))
                yield os.path.join(root, file)

@log_function_call
def convert_position_to_speed():
    # Ottieni la directory corrente
    current_directory = os.getcwd()

    # Trova tutti i file che iniziano con "raw_re_output_" nella directory corrente
    matching_files = [file for file in os.listdir(current_directory) if file.startswith("raw_re_output_")]

    # Itera su ciascun file trovato
    for file in matching_files:
        file_path = os.path.join(current_directory, file)
        print("File trovato:", file_path)
        # Leggi il file CSV
        df = pd.read_csv(file_path)
        # Crea un DataFrame vuoto per gli incrementi di "vx"
        df_incrementi = pd.DataFrame(columns=df.columns)

        # Calcola gli incrementi per ciascuna colonna di "vx"
        for col in df.columns:
            if '_vx' in col:
                incrementi = df[col].diff().abs()  # Calcola gli incrementi
                df_incrementi[col] = incrementi  # Assegna gli incrementi al DataFrame degli incrementi

        # Plot dei valori di "vx" originali
        plt.figure(figsize=(10, 6))
        for col in df.columns:
            if '_vx' in col:
                plt.scatter(df['timestamp'], df[col], label=col, s = 2)
                plt.title("originale")

        # Mantieni solo le colonne relative a "z" nel DataFrame originale
        df_z = df[[col for col in df.columns if '_z' in col]]

        # Combina df_incrementi con le colonne di "z"
        df_combined = pd.concat([df_z, df_incrementi], axis=1)

        # Salva il DataFrame combinato in un nuovo file CSV
        file_path_of = os.path.join(current_directory, "of_"+file)
        df_combined.to_csv(file_path_of, index=False)

        # Plot degli incrementi di "vx"
        plt.figure(figsize=(10, 6))
        for col in df_incrementi.columns:
            plt.scatter(df['timestamp'], df_incrementi[col], label=col + ' Increment', s = 2)

        # Impostazioni dei plot
        plt.xlabel('Timestamp')
        plt.ylabel('Valore di vx')
        plt.title('Valori di vx e relativi incrementi in funzione del timestamp')
        plt.legend()
        plt.grid(True)

        # Mostra i plot
        plt.show()


        #df.to_csv(file_path, index=False)

# Funzione per shiftare un segnale di un certo offset temporale
def shift_signal(signal, timestamps, offset):
    return np.interp(timestamps + offset, timestamps, signal)
def interpole_linear(common_timestamp, y1):
    # Creare una funzione di interpolazione lineare
    f_linear = interp1d(common_timestamp, y1, kind='linear')
    # Generare nuovi punti x per l'interpolazione con un passo fine
    common_timestamp = np.linspace(min(common_timestamp), max(common_timestamp), num=10000)  # 1000 punti per maggiore continuità
    y1 = f_linear(common_timestamp)
    # Verifica che x_new e y_linear abbiano la stessa lunghezza
    assert len(common_timestamp) == len(y1)
    return common_timestamp, y1

@log_function_call
def synchro_data_v_v_e_z(file_raw_optics):
    # Plot di tutti e tre i file CSV
    # plot_increment("data_robot_encoder/1b.csv", label='1b')
    # plot_increment("data_robot_encoder/2b.csv", label='2b')
    # plot_increment("data_robot_encoder/4b.csv", label='4b')

    # # Impostazioni del plot
    # plt.title('Incremento di Traslazione X normalizzato in metri al secondo')
    # plt.xlabel('Tempo [s]')
    # plt.ylabel('Velocità [m/s]')
    # plt.legend()
    # plt.grid(True)
    #
    # # Mostra il grafico
    # plt.show()

    # Leggi i dati dai file CSV
    data_1b = pd.read_csv("data_robot_encoder/1b.csv", delimiter=",")
    data_2b = pd.read_csv("data_robot_encoder/2b.csv", delimiter=",")
    data_4b = pd.read_csv("data_robot_encoder/4b.csv", delimiter=",")

    # Estrai i segnali relativi alla traslazione X e i timestamp
    translation_x_2b = data_2b['/tf/base/tool0_controller/translation/x']
    timestamps_2b = data_2b['__time']

    translation_x_1b = data_1b['/tf/base/tool0_controller/translation/x']
    timestamps_1b = data_1b['__time']

    translation_x_4b = data_4b['/tf/base/tool0_controller/translation/x']
    timestamps_4b = data_4b['__time']

    # Trova il timestamp iniziale del primo segnale
    start_time_1b = timestamps_1b[0]

    # Sottrai il timestamp iniziale dal timestamp del secondo segnale
    timestamps_2b_shifted = timestamps_2b - start_time_1b

    # Sottrai il timestamp iniziale dal timestamp del terzo segnale
    timestamps_4b_shifted = timestamps_4b - start_time_1b

    # Calcola la differenza assoluta istante per istante tra i tre segnali
    difference_signal = np.abs(translation_x_2b - translation_x_1b) + np.abs(translation_x_4b - translation_x_1b)




    FIND_SHIFTER = 0
    if FIND_SHIFTER:
        res_shift = []
        sh_1 = []
        sh_2 = []



        for shift_1 in np.arange(-2,2,0.1):
            for shift_2 in np.arange(-2, 2, 0.1):

                # Shifta i segnali 2 e 4
                shifted_signal_2b = shift_signal(translation_x_2b, timestamps_2b, shift_1)
                shifted_signal_4b = shift_signal(translation_x_4b, timestamps_4b, shift_2)



                # Determina la lunghezza minima tra i due segnali
                min_length = min(len(shifted_signal_2b), len(translation_x_1b), len(shifted_signal_4b))

                # Ritaglia la fine del segnale più lungo per farlo coincidere con la lunghezza del segnale più corto
                shifted_signal_2b = shifted_signal_2b[:min_length]
                shifted_signal_4b = shifted_signal_4b[:min_length]
                translation_x_1b = translation_x_1b[:min_length]

                # Calcola la differenza assoluta istante per istante tra i segnali shiftati
                difference_signal_shifted = np.abs(shifted_signal_2b - translation_x_1b) + np.abs(
                    shifted_signal_4b - translation_x_1b)

                # Calcola la differenza media su tutto il tempo tra i tre segnali
                mean_difference = np.mean(difference_signal_shifted)

                print(mean_difference)
                res_shift.append(mean_difference)
                sh_1.append(shift_1)
                sh_2.append(shift_2)

        serie_valori = [i + 1 for i in range(len(res_shift))]


        # #Plotta la differenza tra i segnali shiftati
        # plt.scatter(serie_valori,res_shift)
        # #plt.scatter(serie_valori,sh_1)
        # #plt.scatter(serie_valori,sh_2)
        # plt.xlabel('Tempo')
        # plt.ylabel('Differenza assoluta')
        # plt.title('Differenza assoluta tra i segnali shiftati')
        # plt.show()

        indice_minimo = res_shift.index(min(res_shift))
        print("shift 1 e 2", sh_1[indice_minimo], sh_2[indice_minimo])
        s11 = sh_1[indice_minimo]
        s22 = sh_2[indice_minimo]

    else:

        s11 = -0.99999999999
        s22 = -0.59999999999

    # Shifta i segnali 2 e 4
    shifted_signal_2b = shift_signal(translation_x_2b, timestamps_2b, s11)
    shifted_signal_4b = shift_signal(translation_x_4b, timestamps_4b, s22)

    # Determina la lunghezza minima tra i due segnali
    min_length = min(len(shifted_signal_2b), len(translation_x_1b), len(shifted_signal_4b))

    # Ritaglia la fine del segnale più lungo per farlo coincidere con la lunghezza del segnale più corto
    shifted_signal_2b = shifted_signal_2b[:min_length]
    shifted_signal_4b = shifted_signal_4b[:min_length]
    translation_x_1b = translation_x_1b[:min_length]

    # Trova la lunghezza minima tra tutti i segnali
    min_length = min(len(translation_x_1b), len(shifted_signal_2b), len(shifted_signal_4b))

    # Calcola il passo temporale originale
    time_step = timestamps_1b.diff().mean()

    # Crea una nuova timestamp basata sul passo temporale originale
    common_timestamp = np.arange(0, min_length * time_step, time_step)

    # Plotta i segnali condividendo lo stesso asse x
    y1 = translation_x_1b[:min_length]
    y2 = shifted_signal_2b[:min_length]
    y3 = shifted_signal_4b[:min_length]
    y1 = savgol_filter(y1, 3, 1)
    y2 = savgol_filter(y2, 3, 1)
    y3 = savgol_filter(y3, 3, 1)

    y1_series = pd.Series(y1)
    y2_series = pd.Series(y2)
    y3_series = pd.Series(y3)

    # Interpolare i valori NaN
    y1_interpolated = y1_series.interpolate(method='linear')
    y2_interpolated = y2_series.interpolate(method='linear')
    y3_interpolated = y3_series.interpolate(method='linear')

    # Calcola la media dei tre segnali interpolati
    posizione_media = np.nanmean(np.vstack([y1_interpolated, y2_interpolated, y3_interpolated]), axis=0)

    velocita_media = np.diff(posizione_media) / np.diff(common_timestamp)

    # Applica una media mobile di 3 elementi al segnale di velocità
    window_size = 20
    velocita_media_smoothed = np.convolve(velocita_media, np.ones(window_size) / window_size, mode='valid')

    # Calcola la nuova lunghezza per il tempo per adattarla alla media mobile
    x_velocita = common_timestamp[1:]  # I timestamp per la velocità sono ridotti di 1 rispetto a x originale
    x_smoothed = x_velocita[window_size - 1:]  #

    velocita_media_smoothed_series = pd.Series(velocita_media_smoothed)
    velocita_media_smoothed_interpolated = velocita_media_smoothed_series.interpolate(method='linear').to_numpy()

    plt.scatter(x_smoothed, velocita_media_smoothed_interpolated, label='Velocità media filtrata', s=3)
    # plt.scatter(common_timestamp, posizione_media, label='Segnale 1', s=1)
    # plt.scatter(common_timestamp, y2, label='Segnale 1', s=1)
    # plt.scatter(common_timestamp, y3, label='Segnale 1', s=1)

    plt.xlabel('Tempo')
    plt.ylabel('Valore del segnale')
    plt.title('Segnali condivisi sull\'asse x')
    plt.legend()
    plt.show()


    #

    return x_smoothed,velocita_media_smoothed_interpolated



def media_mobile(lista, window_size):
    """
    Calcola la media mobile di una lista data_robot_encoder una finestra di dimensione window_size.

    :param lista: La lista di valori.
    :param window_size: La dimensione della finestra per il calcolo della media mobile.
    :return: La lista dei valori della media mobile.
    """
    lista = np.array(lista)
    padding = window_size // 2  # Calcolo del padding necessario per mantenere la stessa lunghezza dell'input
    lista_padded = np.pad(lista, (padding, padding), mode='edge')  # Padding con il primo/ultimo valore per mantenere la stessa lunghezza
    moving_avg = np.convolve(lista_padded, np.ones(window_size) / window_size, mode='valid')
    return moving_avg[:len(lista)]  # Rimuove gli elementi in eccesso per mantenere la stessa lunghezza dell'input


def smoothing(x_fps, marker_n, window_size):
    data = x_fps
    """
      Funzione che suddivide una lista in sottoliste basandosi sul cambio di marker ID e le riassembla mantenendo l'ordine originale.

      Argomenti:
        data_robot_encoder: Lista di dati da suddividere e riassemblare.
        marker_n: Lista di marker ID corrispondenti ai dati.

      Restituisce:
        Lista riassemblata con i dati in ordine originale.
      """

    sublists = []  # Lista per memorizzare le sottoliste
    current_sublist = []  # Lista temporanea per la sottolista corrente
    current_marker = None  # Marker ID corrente

    for i, (datum, marker) in enumerate(zip(data, marker_n)):
        # Controlla il cambio di marker
        if current_marker != marker:

            #print("Spezzato")
            #modifica sublist
            #print(current_sublist)
            #print(current_sublist)
            if len(current_sublist) > window_size:

                print(len(current_sublist))
                current_sublist = media_mobile(current_sublist, window_size)
                print(len(current_sublist))


            sublists.append(current_sublist)
            current_sublist = []
            current_marker = marker

        # Aggiungi il dato alla sottolista corrente
        current_sublist.append(datum)

    # Gestisce l'ultima sottolista (se presente)
    if current_sublist:
        print("Spezzato")
        sublists.append(current_sublist)

    # Riassembla i dati in ordine originale
    reassembled_data = []
    for sublist in sublists:
        reassembled_data.extend(sublist)

    return reassembled_data


def hist_adv(residui):
    # Calcola l'errore sistematico
    errore_sistematico = np.mean(residui)

    # Calcola l'errore casuale
    errore_casuale = np.std(residui)

    # Plotta l'istogramma dei residui
    plt.hist(residui, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.6)

    # Calcola la deviazione standard della distribuzione gaussiana
    sigma_standard = np.std(residui)

    # Crea un array di valori x per la distribuzione gaussiana
    x_gauss = np.linspace(np.min(residui), np.max(residui), 100)

    # Calcola i valori y corrispondenti alla distribuzione gaussiana
    y_gauss = norm.pdf(x_gauss, np.mean(residui), np.std(residui))

    # Plotta la distribuzione gaussiana sopra l'istogramma dei residui
    plt.plot(x_gauss, y_gauss, 'r--', label='Distribuzione Gaussiana')

    # Plotta le linee verticali corrispondenti alla deviazione standard
    plt.axvline(x=errore_sistematico + errore_casuale, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=errore_sistematico - errore_casuale, color='k', linestyle='--', linewidth=1)
    # Aggiungi una linea verticale corrispondente al valore medio dei residui
    plt.axvline(x=np.mean(residui), color='g', linestyle='-', linewidth=3)

    # Aggiungi la deviazione standard nel titolo
    plt.title(f'Istogramma dei Residui\nDeviazione Standard: {sigma_standard:.4f}')

    plt.xlabel('Residui [m]')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_outlier(x,y):
    # Converti le serie Pandas in array NumPy
    x = np.array(x)
    y = np.array(y)

    # Calcola il primo e il terzo quartile di x e y
    Q1_x, Q3_x = np.percentile(x, [10 ,90])
    Q1_y, Q3_y = np.percentile(y, [10, 90])

    # Calcola l'interquartile range di x e y
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y

    # Definisci il range per considerare un valore outlier
    range_outlier = 1.5

    # Trova gli outlier in x
    outlier_x = (x < Q1_x - range_outlier * IQR_x) | (x > Q3_x + range_outlier * IQR_x)

    # Trova gli outlier in y
    outlier_y = (y < Q1_y - range_outlier * IQR_y) | (y > Q3_y + range_outlier * IQR_y)

    # Unisci gli outlier trovati sia in x che in y
    outlier = outlier_x | outlier_y

    # Rimuovi gli outlier da x e y
    x_filtrato = x[~outlier]
    y_filtrato = y[~outlier]

    # Stampa il numero di outlier rimossi
    numero_outlier_rimossi = np.sum(outlier)
    print(f"Hai rimosso {numero_outlier_rimossi} outlier.")
    return x_filtrato , y_filtrato





def save_to_file_OF_results(filename, constant, constant_uncert, velocity):
    with open(filename, 'a') as file:
        file.write(f"{constant},{constant_uncert},{velocity}\n")

# Funzione per fare regressione lineare con incertezza
def weighted_linregress_with_error_on_y(x, y, y_err):
    # Pesi basati sull'errore sull'asse y
    w = 1 / y_err

    # Calcola la media pesata dei valori
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)

    # Calcola le covarianze pesate
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    cov_xx = np.sum(w * (x - x_mean) ** 2)

    # Calcola il coefficiente di regressione pesato e l'intercetta
    slope = cov_xy / cov_xx
    intercept = y_mean - slope * x_mean

    # Calcola l'R^2 considerando solo l'errore sulle y
    residui = y - (slope * x + intercept)
    somma_quadri_residui = np.sum(w * residui ** 2)
    totale = np.sum(w * (y - y_mean) ** 2)
    r_squared = 1 - (somma_quadri_residui / totale)

    return slope, intercept, r_squared


def smart_cutter_df(df, threshold):
    start_idx = 0
    sub_dataframes = []
    for i in range(1, len(df)):
        if df['n_frame'].iloc[i] - df['n_frame'].iloc[i - 1] > threshold:
            # Se c'è una discontinuità
            sub_dataframes.append(df.iloc[start_idx:i])
            start_idx = i
    sub_dataframes.append(df.iloc[start_idx:])
    return sub_dataframes

def delete_static_data_manually(df, marker_riferimento, confidence_delation):
    # Calcola il valore massimo e minimo della posizione x del marker di riferimento
    x_min = df[marker_riferimento].min()
    x_max = df[marker_riferimento].max()

    # Calcola il range del valore di x tenendo conto della confidence delation
    x_range = x_max - x_min
    x_range *= (1 - confidence_delation)

    # Calcola i valori soglia
    x_threshold_min = x_min + confidence_delation * x_range
    x_threshold_max = x_max - confidence_delation * x_range

    # Filtra le righe che non soddisfano i criteri di soglia
    df_filtered = df[(df[marker_riferimento] >= x_threshold_min) & (df[marker_riferimento] <= x_threshold_max)]

    return df_filtered


def imaga_analizer_raw():
    # Directory di partenza
    start_directory = os.getcwd()

    # Directory di acquisizione raw
    acquisition_raw_directory = os.path.join(start_directory, 'aquisition_raw')

    # Itera su tutti i file MP4 nella directory di acquisizione raw e nelle sue sottodirectory
    for mp4_file in iter_mp4_files(acquisition_raw_directory):
        print("File MP4 trovato:", mp4_file)


        cap = cv2.VideoCapture(mp4_file)
        # Fai qualcosa con il file MP4, ad esempio leggilo o elaboralo



        PROC = 1
        # Ottieni il numero totale di frame nel video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Inizializza il contatore dei frame processati
        processed_frames = 0



        n_frames = 0

        if PROC:
            # Elimina il file se già esiste

            header = ['timestamp', '7_z', '7_vx', '8_z', '8_vx', '9_z', '9_vx', '10_z', '10_vx', '11_z', '11_vx']
            df = pd.DataFrame(columns=header)


        while cap.isOpened():
            n_frames = n_frames+1
            row_data = {'n_frame': n_frames}

            #print("_")
            ret, frame = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            row_data = {'timestamp': timestamp}
            if not ret:
                break

            processed_frames += 1

            # Stampa l'avanzamento ad intervalli regolari
            if processed_frames % 100 == 0:  # Stampare ogni 100 frame
                print(f"Frame processati: {processed_frames}/{total_frames}")


            if PROC:
                # Calcola l'altezza dell'immagine
                height = frame.shape[0]

                # Calcola la nuova altezza dopo il ritaglio
                new_height = int(height * 0.3)  # Rimuovi un terzo dell'altezza

                # Esegui il ritaglio dell'immagine
                frame = frame[new_height:, :]




            # Trova i marker ArUco nel frame
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Esempio di regolazione del contrasto e della luminosità
            alpha = 2  # Fattore di contrasto
            beta = 5  # Fattore di luminosità
            gray_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)


            # Esempio di rilevamento dei bordi con Canny edge detector

            if PROC:




                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)





                if ids is not None:
                    # Disegna i marker trovati sul framqe
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Loop attraverso i marker trovati
                    # Loop attraverso i marker trovati
                    #print(len(ids))
                    for i in range(len(ids)):
                        # Calcola la posizione 3D del marker rispetto alla telecamera
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, mtx, dist)
                        marker_position = tvec[0][0]  # Posizione 3D del marker
                        # Salva la posizione del marker
                        x, y, z = marker_position[0], marker_position[1], marker_position[2]

                        # Estrai l'ID de10l marker
                        marker_id = ids[i][0]
                        #print("mkr:ID", marker_id)

                        # if marker_id == 1:
                        #     print("x,y: ",x,y)

                        # Calcola le coordinate x dei corner del marker
                        x_coords = corners[i][0][:, 0]

                        # Calcola la coordinata x approssimativa del centro del marker
                        center_x = np.mean(x_coords)
                        z_key = f'{marker_id}_z'
                        vx_key = f'{marker_id}_vx'

                        row_data[z_key] = z
                        row_data[vx_key] = center_x







            # #cv2.imshow("gray",resize_image(gray_image,50))
            # cv2.imshow("hhh", resize_image(gray_image, 50))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if PROC:
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # Rilascia le risorse
        # Salva il dataframe su un file Excel
        if PROC:
            # Salva il DataFrame in un file CSV
            # Ottieni la directory corrente
            current_directory = os.getcwd()

            # Conta quanti file CSV iniziano con "raw_re_output" nella directory corrente
            count = sum(
                1 for file in os.listdir(current_directory) if file.startswith("raw_re_output") and file.endswith(".csv"))
            df.to_csv('raw_re_output_'+str(count+1)+'.csv', index=False)

        cap.release()
        cv2.destroyAllWindows()

###
def plotter_raw(df):
    # Ciclo attraverso i marker da 1 a 5
    for marker_inr in marker_ids:
        print("marker:", marker_inr)
        # Seleziona solo le righe con valori non nulli per il marker corrente
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]

        # Plot delle coordinate x e z del marker corrente come scatter plot
        plt.scatter(marker_data['n_frame'], marker_data[f'x_ip_{marker_inr}']/2000, label=f'X Coordinate Marker {marker_inr}', color='blue')
        plt.scatter(marker_data['n_frame'], marker_data[f'z_3D_{marker_inr}'], label=f'Z Coordinate Marker {marker_inr}', color='red')
        plt.scatter(marker_data['n_frame'], marker_data[f'x_3D_{marker_inr}'], label=f'X_3D Coordinate Marker {marker_inr}', color='green')

        # Aggiungi etichette e legenda al grafico corrente
        plt.xlabel('Numero Frame')
        plt.ylabel('Coordinate')
        plt.title(f'Coordinate X, X_3D e Z del Marker {marker_inr}')
        plt.legend()

        # Mostra il grafico corrente
        plt.show()


def save_results_to_excel(results, output_excel):
    # Leggi il file Excel esistente, se presente
    if os.path.exists(output_excel):
        df = pd.read_excel(output_excel)
    else:
        df = pd.DataFrame()  # Crea un nuovo DataFrame se il file Excel non esiste

    # Itera sui risultati e aggiungi ogni elemento del dizionario come riga nel DataFrame
    for result in results:
        dict_to_add = {}
        # Estrai il valore per ogni colonna dal dizionario e aggiungi come riga al DataFrame
        for key, value_list in result.items():
            # Verifica se la chiave (header) esiste già nel DataFrame
            if key in df.columns:
                # Se la colonna esiste già, estendi la Serie esistente con i nuovi valori

                dict_to_add[key] = value_list
        df = df.append(pd.DataFrame(dict_to_add))



    # Salva il DataFrame aggiornato nel file Excel
    df.to_excel(output_excel, index=False)
def plotter_raw_analys(df,v_rob):
    results = []  # Lista per memorizzare i risultati

    # Ciclo attraverso i marker da 1 a 5
    for marker_inr in marker_ids:
        #print("marker:", marker_inr)
        # Seleziona solo le righe con valori non nulli per il marker corrente
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]



        all_z_depth = marker_data[f'z_3D_{marker_inr}'].tolist()

        posizione = marker_data[f'x_ip_{marker_inr}'].tolist()


        # Calcola i delta tra i valori di posizione successivi
        opt_flow = [posizione[i + 1] - posizione[i] for i in range(len(posizione) - 1)]


        # Estendi la lista dei delta in modo che abbia la stessa dimensione della lista di input di posizione
        opt_flow.append(opt_flow[-1])  # estendi con l'ultimo valore


        marker_list = [int(marker_inr)] * len(all_z_depth)
        speed_rob_list = [v_rob] * len(all_z_depth)
        # Aggiungi i risultati alla lista
        results.append({'marker': marker_list, 'z_mean': all_z_depth,'vx': opt_flow, 'vx_3D': speed_rob_list})




    return results
#
# #

#imaga_analizer_raw()

#convert_position_to_speed()


# df = pd.read_excel(output_excel)
# marker_rif_delation = 9
# df_filtered = delete_static_data_manually(df, f'x_ip_{marker_rif_delation}', 0.04)
# #plotter_raw(df_filtered)
# win = 20
# multi_df = smart_cutter_df(df_filtered, win)
#
# acc_filter_final = 0.08
#
#
#
#
# if len(multi_df) == 10:
#     ALL_R_2 = []
#     for i in range(len(multi_df)):
#         v_i = i // 2
#         speed_rob = v_all[v_i]
#         print(f"dts = {i}, vct = {speed_rob}")
#
#         df_filtered = delete_static_data_manually(multi_df[i], f'x_ip_{marker_rif_delation}', acc_filter_final)
#         res = plotter_raw_analys(df_filtered,speed_rob)
#         #print(res)
#         save_results_to_excel(res, output_excel_res)
#
#     #
#
# else:
#     print("ERROR CUTTINGGG")

# file_path = 'dati_of/stich_grande.xlsx'
# show_result_ex_file(file_path)

#
x_s, vy_s  =synchro_data_v_v_e_z("results_raw.xlsx")
merge_dataset_extr_int(x_s, vy_s)

#sys.exit()


# file_path_1 = 'dati_of/all_points_big_fix_speed.xlsx'
# show_result_ex_file(file_path_1)
# windowing_vs_uncertanty(file_path_1)
#
#
# constant_analisis()
#



#
#
#
#
#
# rrr = []
# coeff = []
#
# # Calcola il passo tenendo conto della precisione
# passo = 0.002
#
# #Precisione desiderata (numero di cifre decimali significative)
# precisione = 3
# valori = np.arange(0.01, 0.16, passo)
# valori_arrotondati = np.around(valori, decimals=precisione)
# acc_filter_final = 0.075
# for acc_filter in valori_arrotondati:
#
#     coeff.append(acc_filter)
#
#     ALL_R_2 = []
#
#     if len(multi_df) == 10:
#         for i in range(len(multi_df)):
#             print("dataset ",i)
#             df_filtered = delete_static_data_manually(multi_df[i], f'x_ip_{marker_rif_delation}', acc_filter)
#             res = plotter_raw_analys(df_filtered)
#             #print(res)
#             #save_results_to_excel(res, output_excel_res)
#         #
#
#     else:
#         print("ERROR CUTTINGGG")
#
#
#     rrr.append(np.mean(ALL_R_2))
#
#
# # Crea il grafico
# plt.plot(coeff, rrr)
#
# # Aggiungi etichette agli assi
# plt.xlabel('Valori di x')
# plt.ylabel('Valori di y')
#
# # Aggiungi titolo al grafico
# plt.title('Grafico di x vs y')
# # Adatta l'asse y ai dati
# plt.gca().autoscale(axis='y')
#
# # Mostra il grafico
# plt.show()

from utility_library.additional_functions import *
import numpy as np

# Salva il dataframe su un file Excel
output_excel = 'marker_data.xlsx'

###modella tutte le curve e estrai un k (5 k) poi graficali rispetto a vext, e quindi puoi stimare k conoscendo la V
#proseguire con il modello ottico cercando meglio.
#non prendere media e dev std ma servono tutti i valori di velocità per prova e distanzaq

# Definisci il percorso della cartella di calibrazione
folder_calibration = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/CALIBRATION_CAMERA_FILE"
PLOT_SINGLE_PATH = 0
# Carica i parametri di calibrazione della camera
mtx = np.load(os.path.join(folder_calibration, "camera_matrix.npy"))
dist = np.load(os.path.join(folder_calibration, "dist_coeffs.npy"))
print("mtx",mtx)
print("dist", dist)

# Estrai i parametri dalla matrice di calibrazione
fx = mtx[0, 0]  # Lunghezza focale lungo l'asse x

fy = mtx[1, 1]  # Lunghezza focale lungo l'asse y
cx = mtx[0, 2]  # Coordinata x del centro di proiezione
cy = mtx[1, 2]  # Coordinata y del centro di proiezione
print(fx,fy,cx,cy)
video_name = 'GX010118.MP4'
# Definisci il percorso del video
video_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/'+video_name

# Definisci la dimensione del marker ArUco
#MARKER_SIZE = 0.0978 #cm piccolo prova 1
MARKER_SIZE = 0.1557 #cm grande prova 2



# Inizializza il detector di marker ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Cartella per i dati
folder_path = 'dati_marker_optical_flow'
os.makedirs(folder_path, exist_ok=True)

# Percorso del file CSV
csv_path = os.path.join(folder_path, 'dati_marker_optical_flow.csv')

# Apri il video
cap = cv2.VideoCapture(video_path)

# Inizializza il rilevatore di flusso ottico
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Definisci il colore per disegnare i vettori di flusso ottico
color = (0, 255, 0)

marker_ids = [7,8,9,10,11]
v1 = 0.25
v2 = 0.5
v3 = 0.75
v4 = 0.94
v5 = 0.97
v_all = [v1,v2,v3,v4,v5]



# Nome del file Excel
output_excel_res = 'results.xlsx'
if os.path.exists(output_excel_res):

    print("file esiste appendo res")


else:

    # Header del file Excel
    header = ['timestamp','7_z', '7_vx','8_z', '8_vx','9_z', '9_vx','10_z', '10_vx','11_z', '11_vx']

    # Crea un DataFrame vuoto con gli header
    df_res = pd.DataFrame(columns=header)

    # Salva il DataFrame con gli header nel file Excel
    df_res.to_excel(output_excel_res, index=False)

    print(f"Creato il file '{output_excel}' con gli header vuoti.")
# Funzione per il salvataggio dei risultati nel file constant






#
# def modello(x, costante):
#     return costante / x
#
# def compute_dz(Vx, Vx_prime, fx, fy, cx, cy):
#     dz = ((Vx * fx) / Vx_prime )
#
#     return dz
#
#
#
# def windowing_vs_uncertanty(file_path):
#     SHOW_PLOT = 0
#
#     v_ext = []
#     unc_k = []
#     sigma_gauss = []
#     win_size = []
#
#     for window_size in range(1,10,1):
#
#
#         # Elimina il file constant se esiste
#         if os.path.exists("constant.txt"):
#             os.remove("constant.txt")
#
#         # Crea il file constant con gli header
#         with open("constant.txt", 'w') as file:
#             file.write("constant,constant_uncert,velocity\n")
#
#
#         data_robot_encoder = pd.read_excel(file_path)
#         # Rendi positivi i valori di vx e vx_std
#
#         # Rendi positivi i valori di vx e vx_std
#         data_robot_encoder['vx'] = abs(data_robot_encoder['vx'])
#
#
#         # Rimuovi le righe con zeri o valori mancanti nella riga
#         data_robot_encoder = data_robot_encoder[(data_robot_encoder != 0).all(1)]
#
#         # Dividi il DataFrame in base al valore della colonna vx_3D
#         gruppi = data_robot_encoder.groupby('vx_3D')
#
#         # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
#         sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}
#
#         for chiave, valore in sotto_dataframe.items():
#             print(chiave, valore)
#             data_robot_encoder = sotto_dataframe[chiave]
#
#
#
#             # Definisci i colori per i diversi valori di vx_3D
#             color_map = {
#                 v1: 'red',
#                 v2: 'blue',
#                 v3: 'green',
#                 v4: 'orange',
#                 v5: 'purple'
#             }
#             print(fx,fy,cx,cy)
#
#
#
#             x_fps = data_robot_encoder['vx']
#
#
#
#
#
#             marker_n = data_robot_encoder['marker']
#             x = [element * 60 for element in x_fps]
#
#
#             y = data_robot_encoder['z_mean']
#
#             SMOOTHING = 1
#             window = 0
#
#
#             if SMOOTHING:
#                 window = 7
#                 x_or = x
#                 x_s = smoothing(x,marker_n, window_size)
#                 x_s_graph = [x_ii + 1000 for x_ii in x_s]
#                 x = x_s
#
#             #x = media_mobile(x,150)
#
#
#             color_p = color_map[chiave]
#
#             PLOT_OF_RAW  = 1
#             if PLOT_OF_RAW and SMOOTHING:
#
#
#                 x__1 = list(range(len(x)))
#                 #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
#                 plt.plot(x__1, x_or)
#                 plt.plot(x__1, x_s_graph)
#                 marker_aug =  [element * 100 for element in marker_n]
#                 plt.plot(x__1,marker_aug)
#                 if SHOW_PLOT:
#                     plt.show()
#
#
#
#             # Vx_prime_values = np.linspace(min(x), max(x), 100)
#             Vx_prime_values = sorted(x)
#
#
#             # Adatta il modello ai dati
#             parametri, covarianza = curve_fit(modello, x, y)
#
#
#             # Estrai la costante stimata
#             costante_stimata = parametri[0]
#
#             # Calcola l'incertezza associata alla costante
#             incertezza_costante = np.sqrt(np.diag(covarianza))[0]
#
#             # Calcola l'R^2
#             residui = y - modello(x, costante_stimata)
#             somma_quadri_residui = np.sum(residui ** 2)
#             totale = np.sum((y - np.mean(y)) ** 2)
#             r_squared = 1 - (somma_quadri_residui / totale)
#
#             # Calcola i valori del modello per il plotting
#             x_modello = np.linspace(min(x), max(x), 100)
#             y_modello = modello(x_modello, costante_stimata)
#
#             # Salva i dati nel file
#             save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)
#
#             plt.figure(figsize=(15, 10))
#
#             # Grafico dei punti grezzi e del modello
#             plt.scatter(x, y, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
#
#
#
#             #MODELLO GENERICO
#
#
#             plt.plot(x_modello, y_modello, label='Modello genereico Dz = k/OF', color='black',linestyle='-.',)
#
#             plt.xlabel('OF [px/s]')
#             plt.ylabel('Depth [m]')
#             plt.grid(True)
#             plt.ylim(0, 2.1)
#
#             # Plot aggiunto
#             Y_teorico = []
#             for i in range(len(Vx_prime_values)):
#
#
#                 dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
#                 Y_teorico.append(dzi)
#
#             plt.plot(Vx_prime_values, Y_teorico, color="grey",label='Modello teorico Dz = (V_r * fx)/OF')
#
#
#             # Calcola l'errore sistematico
#             residui = (y - Y_teorico) / y
#             errore_sistematico = np.mean(residui)
#
#             # Calcola l'errore casuale
#             errore_casuale = np.std(residui)
#             # Calcola i residui
#
#             costante_teorica = fx * float(chiave)
#
#             plt.title(
#                 f'depth vs Optical flow [z = k / vx] - media mobile filtro :{window}, \n K_th: {costante_teorica:.2f} , K_exp:{costante_stimata:.2f} +- {incertezza_costante:.2f} [px*m]  o [px * m/s] || R^2:{r_squared:.4f} \n Stat on relative residuals (asimptotic - no gaussian): \n epsilon_sistem_REL :  {errore_sistematico*100 :.3f}% , sigma_REL: {errore_casuale*100 :.3f} %')
#
#
#             # Posiziona la legenda in alto a destra
#             plt.legend(loc="upper right")
#
#
#
#
#             # Percorso del file di salvataggio
#             file_path_fig = 'results/speed_'+ str(chiave) +'_k_model.png'
#
#             # Verifica se il file esiste già
#             if os.path.exists(file_path_fig):
#                 # Se il file esiste, eliminilo
#                 os.remove(file_path_fig)
#                 print("removed old plot")
#
#             # Salva la figura
#             plt.savefig(file_path_fig)
#
#
#             if SHOW_PLOT:
#                 plt.show()
#
#             if SHOW_PLOT:
#
#                 hist_adv(residui)
#
#             v_ext.append(color_p)
#             unc_k.append(incertezza_costante)
#             sigma_gauss.append(errore_casuale)
#             win_size.append(window_size)
#
#
#     plt.close('all')
#     # Creazione dei subplot
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
#
#     # Grafico 1: Incertezza associata ai parametri del modello
#     for i in range(len(v_ext)):
#         ax1.scatter(win_size[i], unc_k[i], color=v_ext[i], marker="x", label='Model ' + str(i + 1))
#     ax1.set_xlabel('Window Size [samples]')
#     ax1.set_ylabel('k uncertanty [m*px]')
#
#
#     # Grafico 2: Sigma del modello fittato (Sigma Gauss)
#     for i in range(len(v_ext)):
#         ax2.scatter(win_size[i], sigma_gauss[i], color=v_ext[i], label='Model ' + str(i + 1))
#     ax2.set_xlabel('Window Size [samples]')
#     ax2.set_ylabel('relative sigma of residuals [std]')
#
#     # Imposta il titolo del subplot
#     fig.suptitle('Model Evaluation - moving avarege effect')
#
#     # Mostra il plot
#     plt.show()
#
#
# def show_result_ex_file(file_path):
#     SHOW_PLOT = 1
#
#
#     # Elimina il file constant se esiste
#     if os.path.exists("constant.txt"):
#         os.remove("constant.txt")
#
#     # Crea il file constant con gli header
#     with open("constant.txt", 'w') as file:
#         file.write("constant,constant_uncert,velocity\n")
#
#
#     data_robot_encoder = pd.read_excel(file_path)
#     # Rendi positivi i valori di vx e vx_std
#
#     # Rendi positivi i valori di vx e vx_std
#     data_robot_encoder['vx'] = abs(data_robot_encoder['vx'])
#
#
#     # Rimuovi le righe con zeri o valori mancanti nella riga
#     data_robot_encoder = data_robot_encoder[(data_robot_encoder != 0).all(1)]
#
#     # Dividi il DataFrame in base al valore della colonna vx_3D
#     gruppi = data_robot_encoder.groupby('vx_3D')
#
#     # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
#     sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}
#
#     for chiave, valore in sotto_dataframe.items():
#         print(chiave, valore)
#         data_robot_encoder = sotto_dataframe[chiave]
#
#
#
#         # Definisci i colori per i diversi valori di vx_3D
#         color_map = {
#             v1: 'red',
#             v2: 'blue',
#             v3: 'green',
#             v4: 'orange',
#             v5: 'purple'
#         }
#         print(fx,fy,cx,cy)
#
#
#
#         x_fps = data_robot_encoder['vx']
#
#
#
#
#
#         marker_n = data_robot_encoder['marker']
#         x = [element * 60 for element in x_fps]
#
#
#         y = data_robot_encoder['z_mean']
#
#         SMOOTHING = 0
#         window = 0
#
#
#         if SMOOTHING:
#             window = 7
#             x_or = x
#             x_s = smoothing(x,marker_n, window)
#             x_s_graph = [x_ii + 1000 for x_ii in x_s]
#             x = x_s
#
#         #x = media_mobile(x,150)
#
#
#         color_p = color_map[chiave]
#
#         PLOT_OF_RAW  = 1
#         if PLOT_OF_RAW and SMOOTHING:
#
#
#             x__1 = list(range(len(x)))
#             #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
#             plt.plot(x__1, x_or)
#             plt.plot(x__1, x_s_graph)
#             marker_aug =  [element * 100 for element in marker_n]
#             plt.plot(x__1,marker_aug)
#             if SHOW_PLOT:
#                 plt.show()
#
#
#
#         # Vx_prime_values = np.linspace(min(x), max(x), 100)
#         Vx_prime_values = sorted(x)
#
#
#         # Adatta il modello ai dati
#         parametri, covarianza = curve_fit(modello, x, y)
#
#
#         # Estrai la costante stimata
#         costante_stimata = parametri[0]
#
#         # Calcola l'incertezza associata alla costante
#         incertezza_costante = np.sqrt(np.diag(covarianza))[0]
#
#         # Calcola l'R^2
#         residui = y - modello(x, costante_stimata)
#         somma_quadri_residui = np.sum(residui ** 2)
#         totale = np.sum((y - np.mean(y)) ** 2)
#         r_squared = 1 - (somma_quadri_residui / totale)
#
#         # Calcola i valori del modello per il plotting
#         x_modello = np.linspace(min(x), max(x), 100)
#         y_modello = modello(x_modello, costante_stimata)
#
#         # Salva i dati nel file
#         save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)
#
#         plt.figure(figsize=(15, 10))
#
#         # Grafico dei punti grezzi e del modello
#         plt.scatter(x, y, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
#
#
#
#         #MODELLO GENERICO
#
#
#         plt.plot(x_modello, y_modello, label='Modello genereico Dz = k/OF', color='black',linestyle='-.',)
#
#         plt.xlabel('OF [px/s]')
#         plt.ylabel('Depth [m]')
#         plt.grid(True)
#         plt.ylim(0, 2.1)
#
#         # Plot aggiunto
#         Y_teorico = []
#         for i in range(len(Vx_prime_values)):
#
#
#             dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
#             Y_teorico.append(dzi)
#
#         plt.plot(Vx_prime_values, Y_teorico, color="grey",label='Modello teorico Dz = (V_r * fx)/OF')
#
#
#         # Calcola l'errore sistematico
#         residui = (y - Y_teorico) / y
#         errore_sistematico = np.mean(residui)
#
#         # Calcola l'errore casuale
#         errore_casuale = np.std(residui)
#         # Calcola i residui
#
#         costante_teorica = fx * float(chiave)
#
#         plt.title(
#             f'depth vs Optical flow [z = k / vx] - media mobile filtro :{window}, \n K_th: {costante_teorica:.2f} , K_exp:{costante_stimata:.2f} +- {incertezza_costante:.2f} [px*m]  o [px * m/s] || R^2:{r_squared:.4f} \n Stat on relative residuals (asimptotic - no gaussian): \n epsilon_sistem_REL :  {errore_sistematico*100 :.3f}% , sigma_REL: {errore_casuale*100 :.3f} %')
#
#
#         # Posiziona la legenda in alto a destra
#         plt.legend(loc="upper right")
#
#
#
#
#         # Percorso del file di salvataggio
#         file_path_fig = 'results/speed_'+ str(chiave) +'_k_model.png'
#
#         # Verifica se il file esiste già
#         if os.path.exists(file_path_fig):
#             # Se il file esiste, eliminilo
#             os.remove(file_path_fig)
#             print("removed old plot")
#
#         # Salva la figura
#         plt.savefig(file_path_fig)
#
#
#         if SHOW_PLOT:
#             plt.show()
#
#         if SHOW_PLOT:
#
#             hist_adv(residui)
#
#
# def constant_analisis():
#     # Leggi i dati dal file
#     data_robot_encoder = np.loadtxt("constant.txt", delimiter=',', skiprows=1)
#
#     # Estrai le colonne
#     constant_data = data_robot_encoder[:, 0]
#     constant_uncert_data = data_robot_encoder[:, 1]
#     velocity_data = data_robot_encoder[:, 2]
#
#     # Fai la regressione lineare tenendo conto dell'incertezza sulla costante
#     slope, intercept, r_squared = weighted_linregress_with_error_on_y(velocity_data, constant_data, 1 / constant_uncert_data)
#     # Calcola l'incertezza della pendenza
#     residuals = constant_data - (slope * velocity_data + intercept)
#     uncert_slope = np.sqrt(np.sum(constant_uncert_data ** 2 * residuals ** 2) / np.sum((velocity_data - np.mean(velocity_data)) ** 2))
#     # Calcola l'R^2
#
#     sigma3 = [element * 3 for element in constant_uncert_data]
#
#     plt.figure(figsize=(12, 7))
#     # Grafico
#     plt.scatter(velocity_data, constant_data, label='Dati',s = 15)
#     plt.errorbar(velocity_data, constant_data, yerr=sigma3, fmt='none', label='Incertezza')
#     plt.plot(velocity_data, slope * velocity_data + intercept, color='red', label='k(v_ext) sperimentale')
#     plt.plot(velocity_data, velocity_data* fx, color='orange', label='K(v_ext) teorico')
#     plt.xlabel('V_ext[m/s]')
#     plt.ylabel('Constant [k]')
#     plt.title(
#         f' k_i = f(v_ext) : slope:{slope:.1f} sigma:{uncert_slope:.1f} k/[m/s]|| R^2:{r_squared:.4f} \n incertezza su parametri: {constant_uncert_data[0]:.2f} , {constant_uncert_data[1]:.2f},{constant_uncert_data[2]:.2f},{constant_uncert_data[3]:.2f} [px*m] - 99.7% int')
#     plt.legend()
#     plt.grid(True)
#
#
#
#     # Percorso del file di salvataggio
#     file_path_fig = 'results/k_LR.png'
#
#     # Verifica se il file esiste già
#     if os.path.exists(file_path_fig):
#         # Se il file esiste, eliminilo
#         os.remove(file_path_fig)
#         print("removed old plot")
#
#     # Salva la figura
#     plt.savefig(file_path_fig)
#     plt.show()
#

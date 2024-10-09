import cv2
import math

def process_video(input_path, output_path, skip_start=0, skip_end=0):
    # Apri il file video
    cap = cv2.VideoCapture(input_path)

    # Verifica se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    # Ottieni le proprietà del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configura il VideoWriter per salvare il video di output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puoi cambiare il codec se necessario
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Salta i frame iniziali
    current_frame = 0
    while current_frame < skip_start:
        ret = cap.grab()
        if not ret:
            break
        current_frame += 1

    # Processa i frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Verifica se hai raggiunto i frame da saltare alla fine
        if skip_end > 0 and current_frame >= frame_count - skip_end:
            break

        # Mostra il frame
        cv2.imshow('Frame', frame)
        print(f"Frame {current_frame+1}/{frame_count}")
        # Attendi la pressione di un tasto
        key = cv2.waitKey(0) & 0xFF

        # Definisci i tasti per le azioni
        # 's' per salvare il frame
        # 'd' per scartare il frame
        # 'q' per uscire
        if key == ord('s'):
            # Salva il frame
            out.write(frame)
        elif key == ord('d'):
            # Scarta il frame (non fare nulla)
            pass
        elif key == ord('q'):
            # Esci dal loop
            print("Interruzione del processo.")
            break
        else:
            # Se viene premuto un altro tasto, scarta il frame
            print("Tasto non riconosciuto, il frame sarà scartato.")

        current_frame += 1

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()




def get_video_duration(input_path):
    # Apri il file video
    cap = cv2.VideoCapture(input_path)

    # Verifica se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    # Ottieni il numero totale di frame e il frame rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = 60


    # Controllo per evitare divisione per zero
    if fps == 0:
        print("Impossibile ottenere il frame rate del video.")
        cap.release()
        return

    # Ottieni il timestamp del primo frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Impossibile leggere il primo frame.")
        cap.release()
        return
    timestamp_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Converti in secondi

    # Ottieni il timestamp dell'ultimo frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    if not ret:
        print("Impossibile leggere l'ultimo frame.")
        cap.release()
        return
    timestamp_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Converti in secondi

    # Calcola la durata totale utilizzando i timestamp
    duration = timestamp_end - timestamp_start

    # Stampa i risultati con alta precisione
    print(f"Timestamp iniziale: {timestamp_start:.10f} secondi")
    print(f"Timestamp finale: {timestamp_end:.10f} secondi")
    print(f"Durata totale: {duration:.10f} secondi")

    # Rilascia le risorse
    cap.release()

def calcola_varianza_e_incertezza(dati):
    n = len(dati)
    if n < 2:
        print("Servono almeno due dati per calcolare la varianza.")
        return None, None

    # Calcola la media
    media = sum(dati) / n

    # Calcola le differenze quadrate dalla media
    differenze_quadrate = [(x - media) ** 2 for x in dati]

    # Somma delle differenze quadrate
    somma_diff_quadrate = sum(differenze_quadrate)

    # Calcola la varianza (varianza campionaria usando n - 1)
    varianza = somma_diff_quadrate / (n - 1)

    # Calcola l'incertezza
    incertezza = math.sqrt(varianza / (n - 1))

    print(f"Dati: {dati}")
    print(f"Media: {media}")
    print(f"Varianza: {varianza}")
    print(f"Incertezza: {incertezza}")

    return varianza, incertezza

# Esempio di utilizzo
dati = [59.9766, 60.0100, 59.9933, 59.9933]  # Sostituisci con i tuoi dati
calcola_varianza_e_incertezza(dati)
incertezza = 0.0078 #ms
#process_video('/home/mmt-ben/DepthFromOpticalFlow/GL010131.LRV', 'video_output.mp4', skip_start=10, skip_end=10)
#get_video_duration('/home/mmt-ben/DepthFromOpticalFlow/video_output.mp4')

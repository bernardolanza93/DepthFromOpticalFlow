import functools
import time

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Ottieni il nome della funzione
        function_name = func.__name__
        # Stampa il nome della funzione
        print(f"Calling function: {function_name}")
        # Stampa gli argomenti posizionali
        if args:
            print(f"Positional arguments: {len(args)}")
        # Stampa gli argomenti nominati
        if kwargs:
            print(f"Keyword arguments: {kwargs}")
        # Chiama la funzione originale e ottieni il risultato
        result = func(*args, **kwargs)
        # Restituisci il risultato
        return result
    return wrapper





def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time of {func.__name__}: {processing_time:.6f} seconds")
        return result

    return wrapper


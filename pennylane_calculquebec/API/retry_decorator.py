from time import sleep
import warnings

retries = 10

def retry(retries = 10):
    def decorator(func):    
        def wrapper(*args, **kwargs):
            delay = 0.1
            for i in range(retries):
                try:
                    
                    return func(*args, **kwargs)
                except Exception as e:
                    warnings.warn(f"The request failed. Retrying in {delay} seconds...", stacklevel=2)
                    sleep(delay)
                    delay *= 2
            raise Exception("max retries exceeded for request " + func.__name__)
        return wrapper
    return decorator
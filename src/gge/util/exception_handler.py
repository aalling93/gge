import time  # python in-built module
import logging  # python in-built module


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f"{func.__name__} executed in {end_time - start_time:.3f} seconds."
        )
        return result

    return wrapper


def exception_handler(default_return_value=None):
    """
    A decorator factory to catch exceptions, log them, and return a specified default value.

    Args:
        default_return_value: The value to return in case an exception is caught. Defaults to None.

    Returns:
        A decorator that wraps the function and provides exception handling.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"An error occurred in {func.__name__}: {str(e)}")
                # Return the specified default value in case of an exception.
                return default_return_value

        return wrapper

    return decorator

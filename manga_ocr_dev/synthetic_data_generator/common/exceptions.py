class SkipSample(Exception):
    """A custom exception raised to indicate that a sample should be skipped.

    This exception is used throughout the synthetic data generation pipeline to
    signal that a particular sample cannot be generated successfully and should
    be discarded. This allows the data generation process to continue without
    crashing when it encounters problematic data, such as text that cannot be
    rendered with any available font or an image that fails to compose
    correctly.
    """
    pass
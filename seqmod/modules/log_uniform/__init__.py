

try:
    from .log_uniform import LogUniformSampler
except ModuleNotFoundError:
    # only error when attempt to use the module
    def LogUniformSampler(*args, **kwargs):
        raise ValueError("Seems like you haven't compiled the `log_uniform` "
                         "extension. SampledSoftmax relies on it for speed. "
                         "Please go to /seqmod/modules/log_uniform/ and build "
                         "the extension following the indications")

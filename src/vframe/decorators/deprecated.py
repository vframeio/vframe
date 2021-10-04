###############################################################################
# Helpers for deprecation
###############################################################################

class DeprecationWarning(Warning):  # pylint: disable=redefined-builtin
    """Warning for deprecated calls.
    Since python 2.7 DeprecatedWarning is silent by default. So we define
    our own DeprecatedWarning here so that it is not silent by default.
    """


def warn(msg, category=UserWarning, stacklevel=2):
    """Generate a a warning with stacktrace.
    Parameters
    ----------
    msg : str
        The message of the warning.
    category : class
        The class of the warning to produce.
    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``.
    """
    import warnings
    warnings.warn(msg, category=category, stacklevel=stacklevel)


def warn_deprecated(msg, stacklevel=2):
    """Generate a non-silent deprecation warning with stacktrace.
    The used warning is ``imgaug.imgaug.DeprecationWarning``.
    Parameters
    ----------
    msg : str
        The message of the warning.
    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``
    """
    warn(msg, category=DeprecationWarning, stacklevel=stacklevel)


class deprecated(object):  # pylint: disable=invalid-name
    """Decorator to mark deprecated functions with warning.
    Adapted from
    <https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py>.
    Parameters
    ----------
    alt_func : None or str, optional
        If given, tell user what function to use instead.
    behavior : {'warn', 'raise'}, optional
        Behavior during call to deprecated function: ``warn`` means that the
        user is warned that the function is deprecated; ``raise`` means that
        an error is raised.
    removed_version : None or str, optional
        The package version in which the deprecated function will be removed.
    comment : None or str, optional
        An optional comment that will be appended to the warning message.
    """

    def __init__(self, alt_func=None, behavior="warn", removed_version=None,
                 comment=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version
        self.comment = comment

    def __call__(self, func):
        alt_msg = None
        if self.alt_func is not None:
            alt_msg = "Use ``%s`` instead." % (self.alt_func,)

        rmv_msg = None
        if self.removed_version is not None:
            rmv_msg = "It will be removed in version %s." % (
                self.removed_version,)

        comment_msg = None
        if self.comment is not None and len(self.comment) > 0:
            comment_msg = "%s." % (self.comment.rstrip(". "),)

        addendum = " ".join([submsg
                             for submsg
                             in [alt_msg, rmv_msg, comment_msg]
                             if submsg is not None])

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # getargpec() is deprecated
            # pylint: disable=deprecated-method

            # TODO add class name if class method
            import inspect
            # arg_names = func.__code__.co_varnames

            # getargspec() was deprecated in py3, but doesn't exist in py2
            if hasattr(inspect, "getfullargspec"):
                arg_names = inspect.getfullargspec(func)[0]
            else:
                arg_names = inspect.getargspec(func)[0]

            if "self" in arg_names or "cls" in arg_names:
                main_msg = "Method ``%s.%s()`` is deprecated." % (
                    args[0].__class__.__name__, func.__name__)
            else:
                main_msg = "Function ``%s()`` is deprecated." % (
                    func.__name__,)

            msg = (main_msg + " " + addendum).rstrip(" ").replace("``", "`")

            if self.behavior == "warn":
                warn_deprecated(msg, stacklevel=3)
            elif self.behavior == "raise":
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = "**Deprecated**. " + addendum
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + "\n\n    " + wrapped.__doc__

        return wrapped
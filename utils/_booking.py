import sys
import logging
import warnings

try:
    import logging
    from config.logging_setup_configs import setup_logging
    logger = setup_logging(logger=logging.getLogger(__name__))
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)


class WarnDict(dict):
    """
    A specialized dictionary that issues runtime warnings on key modifications.

    WarnDict is designed to help track modifications in dictionaries that hold
    configuration parameters selection cuts or weights. It warns the user when:

      - A key is overwritten (i.e. setting a value for a key that already exists).
      - A key removal is attempted (either when the key is missing or when it is removed).

    In addition, it provides extra utility methods:

      - report(): Prints a formatted report of the current dictionary contents.
      - convert(): Converts the dictionary items into a list of tuples with each key-value 
                   pair reversed. This is useful for interfacing with external modules that 
                   expect data in that tuple format.

    **Usage Example:**

        >>> cuts = WarnDict()
        >>> cuts["pt_cut"] = "(pt > 20)"
        >>> cuts["eta_cut"] = "(abs(eta) < 2.4)"
        >>> cuts["pt_cut"] = "(pt > 25)"  # Warning: Overwriting existing key: pt_cut
        >>> cuts.pop("non_existing")      # Warning: Key not found: non_existing
        >>> cuts.report()                 # Prints a report of all cuts
        >>> reversed_cuts = cuts.convert()  # Converts and returns reversed tuples

    This class is especially useful when configuration dictionaries are built dynamically
    and inadvertent overwrites or removals need to be flagged during development.
    """

    def __setitem__(self, key, value):
        """
        Set the value for key, warning if the key already exists.

        Parameters:
            key: The key to set.
            value: The value to assign to the key.
        """
        if key in self:
            warnings.warn(f"Overwriting existing key: {key}", stacklevel=2)
        super().__setitem__(key, value)

    def pop(self, key):
        """
        Remove the specified key and return its value, issuing warnings as appropriate.

        Parameters:
            key: The key to remove.

        Returns:
            The value corresponding to the removed key.

        Warnings:
            - If the key is not found, a warning is issued.
            - If the key is found and removed, a warning is also issued.
        """
        if key not in self:
            warnings.warn(f"Key not found: {key}", stacklevel=2)
        if key in self:
            warnings.warn(f"Removing key: {key}", stacklevel=2)
        return super().pop(key)

    def report(self, msg_prefix=None):
        """
        Print a formatted report of the current dictionary contents.

        The report lists all key-value pairs in a structured manner.
        """
        if self.items():
            msg_prefix = "" if msg_prefix is None else f"{msg_prefix}: "
            msg = f"{msg_prefix}Using:\n"
            for key, value in self.items():
                msg += f"    {key}: {value}\n"
            logger.info(msg)

    def convert(self):
        """
        Convert the dictionary into a list of reversed key-value tuples.

        Returns:
            A list of tuples where each tuple is (value, key).

        This conversion can facilitate interfacing with APIs or classes that expect data
        in a (value, key) format.
        """
        return list(map(tuple, (map(reversed, self.items()))))


class Ntuple:
    def __init__(self, path, directory, friends=None, tag=None):
        self.path = path
        self.directory = directory
        if friends is not None:
            self.friends = friends
        self.tag = tag

    def __str__(self):
        if self.tag is None:
            layout = "({}, {})".format(self.path, self.directory)
        else:
            layout = "({}, {}, tag = {})".format(self.path, self.directory, self.tag)
        return layout

    def __eq__(self, other):
        return self.path == other.path and self.directory == other.directory

    def __hash__(self):
        return hash((self.path, self.directory))


class Dataset:
    def __init__(self, name, ntuples, quantities_per_vars=None):
        self.name = name
        self.ntuples = ntuples
        self.quantities_per_vars = quantities_per_vars

    def __str__(self):
        return "Dataset-{}".format(self.name)

    def __repr__(self):
        return self.__str__()

    def add_to_ntuples(self, *new_ntuples):
        for new_ntuple in new_ntuples:
            self.ntuples.append(new_ntuple)

    def __eq__(self, other):
        return self.name == other.name and self.ntuples == other.ntuples

    def __hash__(self):
        return hash((self.name, tuple(self.ntuples)))


class Operation:
    def __init__(self, expression, name):
        self.expression = expression
        self.name = name

    def __eq__(self, other):
        return self.expression == other.expression and self.name == other.name

    def __hash__(self):
        return hash((self.expression, self.name))


class Cut(Operation):
    def __str__(self):
        return "Cut(" + self.name + ", " + self.expression + ")"

    def __repr__(self):
        return self.__str__()


class Weight(Operation):
    def __str__(self):
        return "Weight(" + self.name + ", " + self.expression + ")"

    def __repr__(self):
        return self.__str__()

    def square(self):
        self.name = self.name + "^2"
        self.expression = "({0:})*({0:})".format(self.expression)


class Selection:
    def __init__(self, name=None, cuts=None, weights=None):
        self.name = name
        caller = f"{sys._getframe().f_back.f_code.co_name}"

        self.setup_and_info(cuts, self.set_cuts, caller, "Cuts")
        self.setup_and_info(weights, self.set_weights, caller, "Weights")

    def setup_and_info(self, items, setter, caller, name):
        is_internal = caller.startswith("<") and caller.endswith(">")
        if isinstance(items, WarnDict):
            if not is_internal:
                items.report(f"{caller} ({name})")
            setter(items.convert())
        else:
            if isinstance(items, list) and all(isinstance(item, tuple) for item in items):
                if not is_internal:
                    WarnDict(dict(map(reversed, items))).report(f"{caller}")
            setter(items)

    def __str__(self):
        deb_str = "Selection-{}\n".format(self.name)
        deb_str += "Cuts: {} \n".format(self.cuts)
        deb_str += "Weights: {}\n".format(self.weights)
        return deb_str

    def __eq__(self, other):
        return self.cuts == other.cuts and self.weights == other.weights

    def __hash__(self):
        return hash((tuple(self.cuts), tuple(self.weights)))

    def split(self):
        minimal_selections = list()
        for cut in self.cuts:
            s = Selection(name="-".join([cut.name, self.name]), cuts=[cut])
            minimal_selections.append(s)
        for weight in self.weights:
            s = Selection(name="-".join([weight.name, self.name]), weights=[weight])
            minimal_selections.append(s)
        return minimal_selections

    def add_cut(self, cut_expression, cut_name):
        self.cuts.append(Cut(cut_expression, cut_name))

    def add_weight(self, weight_expression, weight_name):
        self.weights.append(Weight(weight_expression, weight_name))

    def remove_cut(self, cut_name):
        for cut in self.cuts:
            if cut.name is cut_name:
                self.cuts.remove(cut)

    def remove_weight(self, weight_name):
        for weight in self.weights:
            if weight.name is weight_name:
                self.weights.remove(weight)

    def set_cuts(self, cuts):
        self.cuts = list()
        if cuts is not None:
            if isinstance(cuts, list):
                for cut in cuts:
                    if isinstance(cut, Cut):
                        self.cuts.append(cut)
                    elif isinstance(cut, tuple):
                        self.cuts.append(Cut(*cut))
                    else:
                        raise TypeError("not a Cut object or tuple")
            else:
                raise TypeError("a list is needed")

    def set_weights(self, weights):
        self.weights = list()
        if weights is not None:
            if isinstance(weights, list):
                for weight in weights:
                    if isinstance(weight, Weight):
                        self.weights.append(weight)
                    elif isinstance(weight, tuple):
                        self.weights.append(Weight(*weight))
                    else:
                        raise TypeError("not a Weight object or tuple")
            else:
                raise TypeError("a list is needed")


class Action:
    def __init__(self, name, variable):
        self.name = name
        self.variable = variable

    def __str__(self):
        return self.name


class Count(Action):
    pass


class Histogram(Action):
    def __init__(self, name, variable, edges):
        Action.__init__(self, name, variable)
        self.edges = edges

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.variable == other.variable
            and self.edges == other.edges
        )

    def __hash__(self):
        return hash((self.name, self.variable, tuple(self.edges)))

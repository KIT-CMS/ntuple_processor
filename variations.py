from copy import deepcopy

from .booking import Unit
from .booking import dataset_from_crownoutput
from .booking import dataset_from_artusoutput
from .utils import Selection
from .utils import Variation

import logging

logger = logging.getLogger(__name__)


def get_quantities_from_expression(expression):
    # first change all operators to &&
    for operator in ["<", ">", "=", "!=", "+", "-", "*", "/", "||", ">=", "<=", "&&", "(", ")", "!"]:
        expression = expression.replace(operator, "&&")
    # then remove all brackets and spaces
    expression = expression.replace(" ", "")
    # then split by && and remove empty strings and numbers
    quantities = [
        q
        for q in expression.split("&&")
        if not q.replace(".", "", 1).isdigit() and q != ""
    ]
    # special keywords can also be filtered out
    special_keywords = ["true", "false", "abs"]
    quantities = [q for q in quantities if q not in special_keywords]
    # return a set of quantities
    return set(quantities)


class ReplaceVariable(Variation):
    """
    Variation that with the method create makes a deepcopy of
    the selections and actions inside the unit passed as argument and substitutes
    the directory attribute with the according variation.

    Args:
        name (str): name used to identify the instance of
            this class
        variation (str)
    """

    def __init__(self, name, variation):
        Variation.__init__(self, name)
        self.variation = variation

    def create(self, unit):
        new_selections = deepcopy(unit.selections)
        new_actions = deepcopy(unit.actions)
        replaced = False
        if self.variation not in unit.dataset.quantities_per_vars:
            logger.fatal("Variation {} not found in ntuple".format(self.variation))
            raise NameError
        else:
            list_of_quantities = set(unit.dataset.quantities_per_vars[self.variation])
            for sel_obj in new_selections:
                for cut in sel_obj.cuts:
                    # if any quantities used in the cut are in the list of quantities affected by the variation, replace them
                    for quantity in list_of_quantities & get_quantities_from_expression(
                        cut.expression
                    ):
                        cut.expression = cut.expression.replace(
                            quantity,
                            "{quantity}__{var}".format(
                                quantity=quantity, var=self.variation
                            ),
                        )
                        replaced = True
                        logger.debug(
                            f"Replaced expression {quantity} with {cut.expression} ( quantity: {quantity}, var: {self.variation})"
                        )
                for weight in sel_obj.weights:
                    # if any quantities used in the weight are in the list of quantities affected by the variation, replace them
                    for quantity in list_of_quantities & get_quantities_from_expression(
                        weight.expression
                    ):
                        logger.debug(f"Initial weight: {weight.expression}")
                        weight.expression = weight.expression.replace(
                            quantity,
                            "{quantity}__{var}".format(
                                quantity=quantity, var=self.variation
                            ),
                        )
                        replaced = True
                        logger.debug(
                            f"Replaced weight {quantity} with {weight.expression} ( quantity: {quantity}, var: {self.variation})"
                        )
            for act in new_actions:
                # if any quantities used in the action variables are in the list of quantities affected by the variation, replace them
                for quantity in list_of_quantities & get_quantities_from_expression(
                    act.variable
                ):
                    act.variable = act.variable.replace(
                        act.variable,
                        "{quantity}__{var}".format(
                            quantity=act.variable, var=self.variation
                        ),
                    )
                    replaced = True
                    logger.debug(f"Replaced act {quantity} with {act.variable}")
        if not replaced:
            logger.warning(f"For variation {self.variation} on unit {unit} no quantities were replaced, the shift has no effect..")
        return Unit(unit.dataset, new_selections, new_actions, self)


class ChangeDataset(Variation):
    """
    Variation that with the method create makes a deepcopy of
    the dataset inside the unit passed as argument and substitutes
    the directory attribute with folder_name.

    Args:
        name (str): name used to identify the instance of
            this class
        folder_name (str): part of the name of the TDirectoryFile
            in the new dataset following the prefix 'channel_' and
            preceding the suffix '/tree_name', e.g. 'NewFolder' in
            'mt_NewFolder/ntuple'

    Attributes:
        name (str): name used to identify the instance of
            this class
        folder_name (str): part of the name of the TDirectoryFile
            in the new dataset following the prefix 'channel_' and
            preceding the suffix '/tree_name', e.g. 'NewFolder' in
            'mt_NewFolder/ntuple'
    """

    def __init__(self, name, folder_name):
        Variation.__init__(self, name)
        self.folder_name = folder_name

    def create(self, unit):
        # A perfect copy of the daatset is exactly what
        # we need, thus using deepcopy is fine
        def change_folder(ntuple):
            folder, tree = ntuple.directory.split("/")
            ntuple.directory = "{}_{}/{}".format(
                folder.split("_")[0], self.folder_name, tree
            )

        new_dataset = deepcopy(unit.dataset)
        for ntuple in new_dataset.ntuples:
            change_folder(ntuple)
            for friend in ntuple.friends:
                change_folder(friend)
        return Unit(new_dataset, unit.selections, unit.actions, self)


class ReplaceCut(Variation):
    def __init__(self, name, replaced_name, cut):
        Variation.__init__(self, name)
        self.replaced_name = replaced_name
        self.cut = cut

    def create(self, unit):
        # Check that the name is present in at least one of the selections and raise an
        # error if not
        if not set(
            [
                cut.name
                for selection in unit.selections
                for cut in selection.cuts
                if cut.name == self.replaced_name
            ]
        ):
            logger.fatal(
                "Cut {} not found in any selection of this Unit".format(
                    self.replaced_name
                )
            )
            raise NameError
        new_selections = list()
        for selection in unit.selections:
            copy_cuts = list()
            for cut in selection.cuts:
                if cut.name == self.replaced_name:
                    logger.debug(
                        "Substitute {} with {} in selection {}".format(
                            cut, self.cut, selection
                        )
                    )
                    copy_cuts.append(self.cut)
                else:
                    copy_cuts.append(cut)
            new_selections.append(
                Selection(selection.name, copy_cuts, selection.weights)
            )
        return Unit(unit.dataset, new_selections, unit.actions, self)


class ReplaceWeight(Variation):
    def __init__(self, name, replaced_name, weight):
        Variation.__init__(self, name)
        self.replaced_name = replaced_name
        self.weight = weight

    def create(self, unit):
        # Check that the name is present in at least one of the selections and raise an
        # error if not
        if not set(
            [
                weight.name
                for selection in unit.selections
                for weight in selection.weights
                if weight.name == self.replaced_name
            ]
        ):
            logger.fatal(
                "Weight {} not found in any selection of this Unit".format(
                    self.replaced_name
                )
            )
            raise NameError
        new_selections = list()
        for selection in unit.selections:
            copy_weights = list()
            for weight in selection.weights:
                if weight.name == self.replaced_name:
                    logger.debug(
                        "Substitute {} with {} in selection {}".format(
                            weight, self.weight, selection
                        )
                    )
                    copy_weights.append(self.weight)
                else:
                    copy_weights.append(weight)
            new_selections.append(
                Selection(selection.name, selection.cuts, copy_weights)
            )
        return Unit(unit.dataset, new_selections, unit.actions, self)


class RemoveCut(Variation):
    def __init__(self, name, removed_name):
        Variation.__init__(self, name)
        self.removed_name = removed_name

    def create(self, unit):
        # Check that the name is present in at least one of the selections and raise an
        # error if not

        if not set(
            [
                cut.name
                for selection in unit.selections
                for cut in selection.cuts
                if cut.name == self.removed_name
            ]
        ):
            logger.fatal(
                "Cut {} not found in any selection of this Unit".format(
                    self.removed_name
                )
            )
            raise NameError
        new_selections = [deepcopy(selection) for selection in unit.selections]
        for new_selection in new_selections:
            new_selection.remove_cut(self.removed_name)
        return Unit(unit.dataset, new_selections, unit.actions, self)


class RemoveWeight(Variation):
    def __init__(self, name, removed_name):
        Variation.__init__(self, name)
        self.removed_name = removed_name

    def create(self, unit):

        # Check that the name is present in at least one of the selections and raise an
        # error if not
        if not set(
            [
                weight.name
                for selection in unit.selections
                for weight in selection.weights
                if weight.name == self.removed_name
            ]
        ):
            logger.fatal(
                "Weight {} not found in any selection of this Unit".format(
                    self.removed_name
                )
            )
            raise NameError
        new_selections = [deepcopy(selection) for selection in unit.selections]
        for new_selection in new_selections:
            new_selection.remove_weight(self.removed_name)
        return Unit(unit.dataset, new_selections, unit.actions, self)


class AddCut(Variation):
    def __init__(self, name, cut):
        Variation.__init__(self, name)
        self.cut = cut

    def create(self, unit):
        new_selections = [selection for selection in unit.selections]
        new_selections.append(Selection(name=self.cut.name, cuts=[self.cut]))
        return Unit(unit.dataset, new_selections, unit.actions, self)


class AddWeight(Variation):
    def __init__(self, name, weight):
        Variation.__init__(self, name)
        self.weight = weight

    def create(self, unit):
        new_selections = [selection for selection in unit.selections]
        new_selections.append(Selection(name=self.weight.name, weights=[self.weight]))
        return Unit(unit.dataset, new_selections, unit.actions, self)


class SquareWeight(Variation):
    def __init__(self, name, weight_name):
        Variation.__init__(self, name)
        self.weight_name = weight_name

    def create(self, unit):

        # Check that the name is present in at least one of the selections and raise an
        # error if not
        if not set(
            [
                weight.name
                for selection in unit.selections
                for weight in selection.weights
                if weight.name == self.weight_name
            ]
        ):
            logger.fatal(
                "Weight {} not found in any selection of this Unit".format(
                    self.weight_name
                )
            )
            raise NameError
        new_selections = [deepcopy(selection) for selection in unit.selections]
        for new_selection in new_selections:
            for weight in new_selection.weights:
                if weight.name == self.weight_name:
                    weight.square()

        return Unit(unit.dataset, new_selections, unit.actions, self)


class ReplaceCutAndAddWeight(Variation):
    def __init__(self, name, replaced_name, cut, weight):
        Variation.__init__(self, name)
        self.replace_cut = ReplaceCut(name, replaced_name, cut)
        self.add_weight = AddWeight(name, weight)

    def create(self, unit):
        unit = self.replace_cut.create(unit)
        return self.add_weight.create(unit)


class ReplaceMultipleCuts(Variation):
    def __init__(self, name, replaced_names, cuts):
        Variation.__init__(self, name)
        self.replace_cuts = [
            ReplaceCut(name, rep_name, cut)
            for rep_name, cut in zip(replaced_names, cuts)
        ]

    def create(self, unit):
        for replace_cut in self.replace_cuts:
            unit = replace_cut.create(unit)
        return unit


class ReplaceMultipleCutsAndAddWeight(Variation):
    def __init__(self, name, replaced_names, cuts, weight):
        Variation.__init__(self, name)
        self.replace_cuts = [
            ReplaceCut(name, rep_name, cut)
            for rep_name, cut in zip(replaced_names, cuts)
        ]
        self.add_weight = AddWeight(name, weight)

    def create(self, unit):
        for replace_cut in self.replace_cuts:
            unit = replace_cut.create(unit)
        return self.add_weight.create(unit)


class ChangeDatasetReplaceCutAndAddWeight(Variation):
    def __init__(self, name, folder_name, replaced_name, cut, weight):
        Variation.__init__(self, name)
        self.change_dataset = ChangeDataset(name, folder_name)
        self.repl_and_add_weight = ReplaceCutAndAddWeight(
            name, replaced_name, cut, weight
        )

    def create(self, unit):
        unit = self.change_dataset.create(unit)
        return self.repl_and_add_weight.create(unit)


class ChangeDatasetReplaceMultipleCutsAndAddWeight(Variation):
    def __init__(self, name, folder_name, replaced_names, cuts, weight):
        Variation.__init__(self, name)
        self.change_dataset = ChangeDataset(name, folder_name)
        self.repl_and_add_weight = ReplaceMultipleCutsAndAddWeight(
            name, replaced_names, cuts, weight
        )

    def create(self, unit):
        unit = self.change_dataset.create(unit)
        return self.repl_and_add_weight.create(unit)


class ReplaceVariableReplaceCut(Variation):
    def __init__(self, name, variation, replaced_name, cut):
        Variation.__init__(self, name)
        self.replace_variable = ReplaceVariable(name, variation)
        self.replace_cut = ReplaceCut(name, replaced_name, cut)

    def create(self, unit):
        unit = self.replace_variable.create(unit)
        return self.replace_cut.create(unit)


class ReplaceVariableReplaceCutAndAddWeight(Variation):
    def __init__(self, name, variation, replaced_name, cut, weight):
        Variation.__init__(self, name)
        self.replace_variable = ReplaceVariable(name, variation)
        self.repl_and_add_weight = ReplaceCutAndAddWeight(
            name, replaced_name, cut, weight
        )

    def create(self, unit):
        unit = self.replace_variable.create(unit)
        return self.repl_and_add_weight.create(unit)

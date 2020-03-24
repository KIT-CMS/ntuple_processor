from .utils import Dataset
from .utils import Selection
from .utils import Friend
from .utils import Ntuple
from .utils import Cut
from .utils import Weight
from .utils import Action
from .utils import Count
from .utils import Histogram

from ROOT import gROOT
gROOT.SetBatch(True)
from ROOT import TFile

import os
import re
import json

import logging
logger = logging.getLogger(__name__)


class DatasetFromNameSet:
    """Fake class introduced to simulate a static behavior for
    the dataset, in order for it to be created only once.
    The function can be called with the name 'dataset_from_nameset'
    from the API.

    Attributes:
        dataset (Dataset): Dataset object created with the function
        inner_dataset_from_nameset
    """
    def __init__(self):
        self._cache = {}

    def __call__(
            self,
            dataset_name,
            file_names, folder,
            files_base_directories,
            friends_base_directories):
        if dataset_name not in self._cache:
            self._cache[dataset_name] = self.inner_dataset_from_nameset(
                    dataset_name,
                    file_names, folder,
                    files_base_directories,
                    friends_base_directories)
        return self._cache[dataset_name]

    def inner_dataset_from_nameset(
            self,
            dataset_name,
            file_names, folder,
            files_base_directories,
            friends_base_directories):
        """Create a Dataset object from a list containing the names
        of the ROOT files (e.g. [root_file1, root_file2, (...)]):
            ntuple1: /file_base_dir/root_file1/folder/ntuple
                friend1: /friend1_base_dir/root_file1/folder/ntuple
                friend2: /friend2_base_dir/root_file1/folder/ntuple
            ntuple2: /file_base_dir/root_file2/folder/ntuple
                friend1: /friend1_base_dir/root_file2/folder/ntuple
                friend2: /friend2_base_dir/root_file2/folder/ntuple
            ntuple3: /file_base_dir/root_file3/folder/ntuple
                friend1: /friend1_base_dir/root_file3/folder/ntuple
                friend2: /friend2_base_dir/root_file3/folder/ntuple
            (...)

        Args:
            dataset_name (str): Name of the dataset
            file_names (list): List containing the names of the .root
                files
            folder (str): Name of the TDirectoryFile
            files_base_directories (str, list): Path (list of paths) to
                the files base directory (directories)
            friends_base_directories (str, list): Path (list of paths) to
                the friends base directory (directories)

        Returns:
            dataset (Dataset): Dataset object containing TTrees
        """
        def get_complete_filenames(directory, files):
            full_paths = []
            for f in files:
                full_paths.append(
                    os.path.join(
                        directory, f, "{}.root".format(f)
                        )
                    )
            return full_paths

        def get_full_tree_name(
                folder, path_to_root_file, tree_name):
            root_file = TFile(path_to_root_file)
            if root_file.IsZombie():
                logger.fatal(
                    'File {} does not exist, abort'.format(
                        path_to_root_file))
                raise FileNotFoundError(
                    'File {} does not exist, abort'.format(
                        path_to_root_file))
            else:
                if folder not in root_file.GetListOfKeys():
                    raise NameError(
                        'Folder {} not in {}\n'.format(folder, path_to_root_file))
                full_tree_name = '/'.join([folder, tree_name])
                return full_tree_name

        # E.g.: file_base_dir/file_name.root
        root_files = get_complete_filenames(
            files_base_directories, file_names)
        logger.debug('%%%%%%%%%% Creating dataset {}'.format(dataset_name))
        ntuples = []

        # E.g.: file_base_dir/file_name1.root/folder/ntuple
        #       file_base_dir/file_name2.root/folder/ntuple
        for (root_file, name) in zip(root_files, file_names):
            tdf_tree = get_full_tree_name(
                folder, root_file, 'ntuple')
            if tdf_tree:
                friends = []
                friend_paths = []
                for friends_base_directory in friends_base_directories:
                    friend_paths.append(os.path.join(
                        friends_base_directory, name, "{}.root".format(name)))
                for friend_path in friend_paths:
                    if os.path.isfile(friend_path):
                        friends.append(Friend(friend_path, tdf_tree))
                    else:
                        logger.fatal(
                            'File {} does not exist, abort'.format(
                                friend_path))
                        raise FileNotFoundError(
                            'File {} does not exist, abort'.format(
                                friend_path))
                ntuples.append(Ntuple(root_file, tdf_tree, friends))
        dataset = Dataset(dataset_name, ntuples,
                file_names,
                folder,
                files_base_directories,
                friends_base_directories)

        return dataset

dataset_from_nameset = DatasetFromNameSet()

class Unit:
    """
    Building element of a minimal analysis flow, consisting
    of a dataset, a set of selections to apply on the data
    and a set of actions.

    Args:
        dataset (Dataset): Set of TTree objects to run the
            analysis on
        selections (list): List of Selection-type objects
        actions (Action): Actions to perform on the processed
            dataset, can be 'Histogram' or 'Count'
        variation (Variation): Variations applied, meaning
            that this selection is the result of a variation
            applied on other selections

    Attributes:
        dataset (Dataset): Set of TTree objects to run the
            analysis on
        selections (list): List of Selection-type objects
        actions (Action): Actions to perform on the processed
            dataset, can be 'Histogram' or 'Count'
        variation (Variation): Variations applied, meaning
            that this selection is the result of a variation
            applied on other selections
    """
    def __init__(
            self,
            dataset, selections, actions,
            variation = None):
        self.__set_dataset(dataset)
        self.__set_selections(selections)
        self.__set_actions(actions)
        if variation is not None:
            self.__set_variation(variation)

    def __str__(self):
        layout = '\n'.join([
            'Dataset: {}'.format(self.dataset.name),
            'Selections: {}'.format(self.selections),
            'Actions: {}'.format(self.actions)])
        return layout

    def __set_dataset(self, dataset):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'TypeError: not a Dataset object.')

    def __set_selections(self, selections):
        if isinstance(selections, list):
            is_selection = True
            for selection in selections:
                if not isinstance(selection, Selection):
                    is_selection = False
                    break
            if is_selection:
                self.selections = selections
            else:
                raise TypeError(
                   'TypeError: not Selection objects.')
        else:
            raise TypeError(
                'TypeError: not a list object.')

    def __set_actions(self, actions):
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, Action):
                    is_action = True
                else:
                    is_action = False
                    break
            if is_action:
                self.actions = [self.__set_new_action(action) \
                        for action in actions]
            else:
                raise TypeError(
                   'TypeError: not Action objects.')
        else:
            raise TypeError(
                    'TypeError: not a list object.')

    def __set_new_action(self, action):
        name = '#'.join([action.variable,
            self.dataset.name,
            '-'.join([selection.name for selection in self.selections])])
        if isinstance(action, Histogram):
            name = '#'.join([name, action.binning.name])
            return Histogram(
                    action.variable, action.binning.edges,
                    name)
        elif isinstance(action, Count):
            return Count(action.variable, name)

    def __set_variation(self, variation):
        self.variation = variation
        for action in self.actions:
            action.name = '#'.join([action.name, self.variation.name])

    def __eq__(self, other):
        return self.dataset == other.dataset and \
            self.selections == other.selections and \
            self.actions == other.actions

    def __hash__(self):
        return hash((
            self.dataset, tuple(self.selections),
            tuple(self.actions)))

class UnitManager:
    """
    Manager of all the Unit objects that are created.
    It can both be initialized with a variable amount of Unit
    objects as arguments or with no arguments, with the above mentioned
    objects added in a second time with the function 'book'.

    Attributes:
        booked_units (list): List of the booked units, updated during
            initialization or with the function 'book'
    """

    booked_units = []

    def book(self, units, variations = None):
        for unit in units:
            if unit not in self.booked_units:
                self.booked_units.append(unit)
        if variations:
            for variation in variations:
                logger.debug('Applying variation {}'.format(variation))
                for unit in units:
                    self.apply_variation(unit, variation)

    def apply_variation(self, unit, variation):
        new_unit = variation.create(unit)
        self.booked_units.append(new_unit)

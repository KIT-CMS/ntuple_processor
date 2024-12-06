from .utils import Dataset
from .utils import Selection
from .utils import Ntuple
from .utils import Cut
from .utils import Weight
from .utils import Action
from .utils import Count
from .utils import Histogram
from .utils import Variation

from ROOT import gROOT

gROOT.SetBatch(True)
from ROOT import TFile

import os
import re
import yaml
import itertools
from XRootD import client

import logging

logger = logging.getLogger(__name__)


def dataset_from_artusoutput(
    dataset_name, file_names, folder, files_base_directory, friends_base_directories
):
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
        folder (str): Name of the TDirectoryFile in each .root file
        files_base_directory (str): Path to the files base directory (directories)
        friends_base_directories (str, list): List of paths to
            the friends base directory (directories)

    Returns:
        dataset (Dataset): Dataset object containing TTrees
    """

    def get_full_tree_name(folder, path_to_root_file, tree_name):
        root_file = TFile.Open(path_to_root_file)
        if root_file.IsZombie():
            logger.fatal("File {} does not exist, abort".format(path_to_root_file))
            raise FileNotFoundError
        if folder not in root_file.GetListOfKeys():
            logger.fatal(
                "Folder {} does not exist in {}\n".format(folder, path_to_root_file)
            )
            raise NameError
        root_file.Close()
        full_tree_name = "/".join([folder, tree_name])
        return full_tree_name

    def add_tagged_friends(friends):
        """Tag friends with the name of the different directories
        in the artus name scheme, e.g.:
        /common_path/MELA/ntuple -> tag: MELA
        /common_path/SVFit/ntuple -> tag: SVFit
        Since when we compare two ntuples (with full path) only one
        directory changes in this scheme (see MELA vs SVFit), we
        create a list called 'tags' with these two strings; then we
        assign this string to friend.tag, if it's None
        """
        for f1, f2 in itertools.combinations(friends, 2):
            l1 = f1.path.split("/")
            l2 = f2.path.split("/")
            tags = list(set(l1).symmetric_difference(set(l2)))
            if tags:
                for t in tags:
                    if t in l1 and f1.tag is None:
                        f1.tag = t
                    elif t in l2 and f2.tag is None:
                        f2.tag = t
        return friends

    # E.g.: file_base_dir/file_name/file_name.root
    root_files = [
        os.path.join(files_base_directory, f, "{}.root".format(f)) for f in file_names
    ]

    # E.g.: file_base_dir/file_name1/file_name1.root/folder/ntuple
    #       file_base_dir/file_name1/file_name2.root/folder/ntuple
    ntuples = []
    for root_file, file_name in zip(root_files, file_names):
        tdf_tree = get_full_tree_name(folder, root_file, "ntuple")
        friends = []
        for friends_base_directory in friends_base_directories:
            friend_path = os.path.join(
                friends_base_directory, file_name, "{}.root".format(file_name)
            )
            tdf_tree_friend = get_full_tree_name(folder, friend_path, "ntuple")
            if tdf_tree != tdf_tree_friend:
                logger.fatal(
                    "Extracted wrong TDirectoryFile from friend which is not the same than the base file."
                )
                raise Exception
            friends.append(Ntuple(friend_path, tdf_tree_friend))
        ntuples.append(Ntuple(root_file, tdf_tree, add_tagged_friends(friends)))

    return Dataset(dataset_name, ntuples)


def dataset_from_crownoutput(
    dataset_name,
    file_names,
    era,
    channel,
    folder,
    files_base_directory,
    friends_base_directories=None,
    validate_samples=False,
    validation_tag="v1",
    xrootd=False,
):
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
        channel (str): Name of the considered channel, needed for directories
        folder (str): Name of the TDirectoryFile in each .root file
        files_base_directory (str): Path to the files base directory (directories)
        friends_base_directories (str, list): List of paths to
            the friends base directory (directories)

    Returns:
        dataset (Dataset): Dataset object containing TTrees
    """

    def get_quantities_per_variation(root_file):
        quantities_per_vars = {}
        quantities_with_variations = root_file.Get("ntuple").GetListOfLeaves()
        for qwv in quantities_with_variations:
            qwv_name = qwv.GetName()
            if "__" in qwv_name:
                quantity, var = qwv_name.split("__")
                if var not in quantities_per_vars.keys():
                    quantities_per_vars[var] = []
                quantities_per_vars[var].append(quantity)
            if "-" in qwv_name:
                logger.warning(
                    "Found a '-' in quantity name {} - This can result in unwanted behaviour for systematic shifts".format(
                        qwv_name
                    )
                )
        return quantities_per_vars
    def is_root_file_empty(file_path):
        try:
            # Try to open the ROOT file
            root_file = TFile.Open(file_path)

            # Check if the file was opened successfully
            if not root_file or root_file.IsZombie():
                print(f"Failed to open file '{file_path}' or it is a zombie.")
                return True

            # Check if the file contains any keys
            if root_file.GetNkeys() == 0:
                print(f"File '{file_path}' contains no keys.")
                root_file.Close()
                return True
            if "ntuple" not in [x.GetTitle() for x in root_file.GetListOfKeys()]:
                return True

            # If we reach here, the file is not empty
            root_file.Close()
            # print(f"File '{file_path}' is not empty.")
            return False

        except Exception as e:
            print(f"An error occurred while opening the file: {e}")
            return True
        
    def add_tagged_friends(friends):
        """Tag friends with the name of the different directories
        in the CROWN name scheme, e.g.:
        /common_path/CROWNFriends/fastmtt/ntuple -> tag: fastmtt
        /common_path/CROWNMultiFriends/NNclassification/ntuple -> tag: NNclassification
        Since using CROWN two folders containing friends could be present, `CROWNFriends`
        and `CROWNMultiFriends`, both have to be checked. The next folder in the hierarchy
        gives us the friend tag independent of friend or multifriend.
        """
        for friend in friends:
            path_split_list = friend.path.split("/")
            if "CROWNFriends" in friend.path:
                idx = path_split_list.index("CROWNFriends")
            elif "CROWNMultiFriends" in friend.path:
                idx = path_split_list.index("CROWNMultiFriends")
            friend_tag = path_split_list[idx+1]
            if friend.tag is None:
                friend.tag = friend_tag
        return friends

    def populate_val_database(root_file_path, validation_dict, friends):
        def extract_quantities(infile):
            if infile.IsZombie():
                logger.fatal("File {} does not exist, abort".format(infile.GetName()))
                raise FileNotFoundError
            if "ntuple" not in [x.GetTitle() for x in infile.GetListOfKeys()]:
                quantities = set()
            else:
                quantities = set(
                    [x.GetName() for x in infile.Get("ntuple").GetListOfLeaves()]
                )
            return quantities

        # Check if file can be opened and is not empty
        # Check if file can be opened and is not empty
        quantities = {}
        if is_root_file_empty(root_file_path):
            #validation_dict["quantities_per_vars"] = {}
            is_empty = True
        else:
            is_empty = False
            root_file = TFile.Open(root_file_path)
            quantities = extract_quantities(root_file)
        if "quantities_per_vars" in validation_dict or is_empty:
            pass
        else:
            validation_dict["quantities_per_vars"] = get_quantities_per_variation(
                root_file
            )
            root_file.Close()
        # Do the same for the friend trees
        friend_quantitites = set()
        friend_empty = False
        for f in friends:
            if is_root_file_empty(f) or is_empty:
                friend_empty = True
            else:
                friend = TFile.Open(f)
                fr_quants = extract_quantities(friend)
                friend.Close()
                friend_quantitites.update(fr_quants)
        # first we check the main ntuple, then the friends
        fileinfo = {}
        fileinfo["is_empty"] = is_empty
        fileinfo["friend_is_empty"] = friend_empty
        if len(validation_dict["varset"]) == 0:
            validation_dict["varset"] = quantities
            difference = set()
        else:
            difference = validation_dict["varset"].symmetric_difference(quantities)
        fileinfo["difference"] = difference
        if len(validation_dict["friends_varset"]) == 0:
            validation_dict["friends_varset"] = friend_quantitites
            difference = set()
        else:
            difference = validation_dict["friends_varset"].symmetric_difference(
                friend_quantitites
            )
        fileinfo["friends"] = friends
        fileinfo["friends_difference"] = difference
        validation_dict["files"][root_file_path] = fileinfo
        return

    # files_base_directory: ntuple/era
    # friends_base_directory: friends/friend_type/era
    root_files = []
    # Set up reading of file system via xrootd bindings
    if xrootd:
        fsname = "root://cmsdcache-kit-disk.gridka.de:1094"
        xrdclient = client.FileSystem(fsname)
        for f in file_names:
            status, listing = xrdclient.dirlist(
                os.path.join("", files_base_directory, era, f, channel)
            )
            try:
                for g in listing:
                    # os.path.join omits parts with colons as it thinks they are drives,
                    # use default join instead
                    filepath = "/".join([fsname,os.path.join(files_base_directory, era, f, channel, g.name),])
                    if filepath.endswith(".root"):
                        root_files.append((filepath, f))
            except TypeError:
                logger.error(
                    "Could not read file list from directory {}".format(
                        os.path.join(files_base_directory, era, f, channel)
                    )
                )
                raise TypeError
    else:
        for f in file_names:
            for g in os.listdir(os.path.join(files_base_directory, era, f, channel)):
                # only consider files with .root in the end
                filepath = (os.path.join(files_base_directory, era, f, channel, g), f)
                if filepath[0].endswith(".root"):
                    root_files.append(filepath)
    ntuples = []
    read_from_database = False
    db_path = os.path.join("validation_database",validation_tag)
    if os.path.exists(f"{db_path}/{era}_{channel}_{dataset_name}.yaml"):
        logger.info(
            "Reading validation information for dataset {} - {} - {}".format(
                era, channel, dataset_name
            )
        )
        with open(f"{db_path}/{era}_{channel}_{dataset_name}.yaml") as fi:
            validation_dict = yaml.safe_load(fi)
        read_from_database = True
    else:
        logger.info(
            "Running ntuple validation for {} - {} - {}".format(
                era, channel, dataset_name
            )
        )
        validation_dict = {"varset": set(), "friends_varset": set(), "files": {}}
    for root_file, file_name in root_files:
        tdf_tree = "ntuple"
        friends = []
        friend_paths = []
        for friends_base_directory in friends_base_directories:
            friend_base_name = os.path.basename(root_file)
            if xrootd:
                friend_path = "/".join(
                    [
                        fsname,
                        os.path.join(
                            friends_base_directory,
                            era,
                            file_name,
                            channel,
                            friend_base_name,
                        ),
                    ]
                )
            else:
                friend_path = os.path.join(
                    friends_base_directory, era, file_name, channel, friend_base_name
                )
            friend_paths.append(friend_path)
        if not read_from_database:
            populate_val_database(root_file, validation_dict, friend_paths)
        if root_file not in validation_dict["files"]:
            raise ValueError(
                "File {} not found in validation results.".format(root_file)
            )
        if not validation_dict["files"][root_file]["friend_is_empty"]:
            for friend_path in friend_paths:
                friends.append(Ntuple(friend_path, tdf_tree))
        if not validation_dict["files"][root_file]["is_empty"]:
            ntuples.append(Ntuple(root_file, tdf_tree, add_tagged_friends(friends)))
    # Perform check of number of files
    if len(validation_dict["files"].keys()) > len(root_files):
        miss_files = set(validation_dict["files"].keys()).difference(
            set(rfile[0] for rfile in root_files)
        )
        logger.fatal(
            f"Number of files expected and in database do not agree for dataset {dataset_name}.\n"
            f"The missing files are: {miss_files}"
        )
        raise ValueError(
            f"Number of files expected and in database do not agree for dataset {dataset_name}."
        )
    # Report on obtained validation information for dataset
    found_error = False
    for root_file, _ in root_files:
        diff = validation_dict["files"][root_file]["difference"]
        friends_diff = validation_dict["files"][root_file]["friends_difference"]
        if len(diff) != 0 or len(friends_diff) != 0:
            found_error = True
            logger.fatal(
                "Validation for {} - {} - {} failed, differences were found".format(
                    era, channel, dataset_name
                )
            )
            if len(diff) != 0:
                logger.fatal("File {} has the following differences:".format(root_file))
                logger.fatal("\t{}".format(diff))
            if len(friends_diff) != 0:
                logger.fatal(
                    "Friends for file {} have the following differences:".format(
                        root_file
                    )
                )
                logger.fatal("\t{}".format(friends_diff))
    if not found_error:
        logger.info(
            "Validation for {} - {} - {} passed".format(era, channel, dataset_name)
        )
    # Write the created database
    if not read_from_database:
        db_path = os.path.join("validation_database",validation_tag)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        logger.info(
            "Writing validation info for {e} - {c} - {dn} to {db_path}/{e}_{c}_{dn}.yaml".format(
                e=era, c=channel, dn=dataset_name, db_path=db_path
            )
        )
        with open(
            f"{db_path}/{era}_{channel}_{dataset_name}.yaml", "w"
        ) as outfi:
            yaml.safe_dump(validation_dict, outfi, sort_keys=True, indent=4)
    return Dataset(
        dataset_name,
        ntuples,
        quantities_per_vars=validation_dict["quantities_per_vars"],
    )


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

    def __init__(self, dataset, selections, actions, variation=None):
        self.__set_dataset(dataset)
        self.__set_selections(selections)
        self.__set_actions(actions, variation)

    def __str__(self):
        layout = "\n".join(
            [
                "Dataset: {}".format(self.dataset.name),
                "Selections: {}".format(self.selections),
                "Actions: {}".format(self.actions),
            ]
        )
        return layout

    def __set_dataset(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("not a Dataset object.")
        self.dataset = dataset

    def __set_selections(self, selections):
        if not isinstance(selections, list):
            raise TypeError("not a list object.")
        for selection in selections:
            if not isinstance(selection, Selection):
                raise TypeError("not a Selection object.")
        self.selections = selections

    def __set_actions(self, actions, variation):
        if not isinstance(actions, list):
            raise TypeError("not a list object.")
        for action in actions:
            if not isinstance(action, Action):
                raise TypeError("not an Action object.")
        self.actions = [self.__set_new_action(action, variation) for action in actions]

    def __set_new_action(self, action, variation):
        if variation is None:
            name = "#".join(
                [
                    self.dataset.name,
                    "-".join([selection.name for selection in self.selections]),
                    "Nominal",
                    action.name,
                ]
            )
        else:
            if not isinstance(variation, Variation):
                raise TypeError("not a Variation object.")
            self.variation = variation
            name = action.name.replace("Nominal", self.variation.name)
        if isinstance(action, Histogram):
            return Histogram(name, action.variable, action.edges)
        elif isinstance(action, Count):
            return Count(name, action.variable)

    def __eq__(self, other):
        return (
            self.dataset == other.dataset
            and self.selections == other.selections
            and self.actions == other.actions
        )

    def __hash__(self):
        return hash((self.dataset, tuple(self.selections), tuple(self.actions)))


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

    def book(self, units, variations=None, enable_check=True):
        for unit in units:
            if unit not in self.booked_units:
                self.booked_units.append(unit)
        if variations:
            for variation in variations:
                logger.debug("Applying variation {}".format(variation))
                for unit in units:
                    self.apply_variation(unit, variation)
        if enable_check:
            for action1, action2 in itertools.combinations(
                [j for i in [unit.actions for unit in self.booked_units] for j in i], 2
            ):
                if action1.name == action2.name:
                    logger.fatal(
                        "Caught two actions with same name ({}, {})".format(
                            action1.name, action2.name
                        )
                    )
                    raise NameError

    def apply_variation(self, unit, variation):
        new_unit = variation.create(unit)
        self.booked_units.append(new_unit)

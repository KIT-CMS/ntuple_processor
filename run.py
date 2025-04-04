import time
import re
from collections import defaultdict
from multiprocessing import Pool

from .utils import Count
from .utils import Histogram
from .utils import RDataFrameCutWeight

from ROOT import gROOT

gROOT.SetBatch(True)
from ROOT import RDataFrame
from ROOT import TFile
from ROOT import TChain
from ROOT import EnableImplicitMT
from ROOT.std import vector

try:
    import logging
    from config.logging_setup_configs import setup_logging
    logger = setup_logging(logger=logging.getLogger(__name__))
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)


class NestedDefaultDict(defaultdict):
    """
    A nested defaultdict that allows for easy creation of
    multi-level dictionaries with default values.
    This class is a subclass of defaultdict and is used to
    create a nested dictionary structure where each level
    is also a defaultdict.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))

    @property
    def regular(self) -> dict:
        """
        Convert the defaultdict to a regular dictionary, useful i.e. when saving as yaml
        """
        def convert(d):
            if isinstance(d, defaultdict):
                d = {k: convert(v) for k, v in d.items()}
            return d
        return convert(self)


class RunManager:
    """Convert the Graph-style language into PyROOT/RDataFrame
    language and schedule RDataFrame operations, like the
    following:
        Dataset()     -->   RDataFrame()
        Selection()   -->   Filter()
        Count()       -->   Sum()
        Histogram()   -->   Histo1D()

    Args:
        graphs (list): List of Graph objects that are converted
            node by node to RDataFrame operations

    Attributes:
        graphs (list): List of graphs to be processed
        tchains (list): List of TChains created, saved as attribute
            for the class in order to not let them go out of scope
        friend_tchains (list): List of friend TChains created,
            saved as attribute for the class in otder to not let
            them out of scope
    """

    def __init__(self, graphs, *, create_histograms=True, create_config=False):
        self.graphs = graphs
        self.tchains = list()
        self.friend_tchains = list()

        self.create_histograms = create_histograms
        self.create_config = create_config
        if create_config:
            self.config = NestedDefaultDict()

    def _run_multiprocess(self, graph):
        start = time.time()
        ptrs, diags = self.node_to_root(graph)
        logger.info(
            "Event loop for graph {:} started at {} (Number of shapes {})".format(
                repr(graph), time.strftime("%H:%M:%S", time.gmtime(time.time())), len(ptrs)
            )
        )
        logger.debug(
            "%%%%%%%%%% Ready to produce a subset of {} shapes".format(len(ptrs))
        )
        if logger.getEffectiveLevel() == logging.DEBUG:
            reports = [node.Report() for node in diags]
        results = list()
        for ptr in ptrs:
            th = ptr.GetValue()
            results.append(th)
        # Sanity check: event loop run only once for each RDataFrame
        for diag_node in diags:
            loops = diag_node.GetNRuns()
            if loops != 1:
                logger.warning("Event loop run {} times".format(loops))

        if logger.getEffectiveLevel() == logging.DEBUG:
            for report in reports:
                report.Print()

        end = time.time()
        logger.info(
            "Event loop for graph {:} run in {:.2f} seconds".format(
                repr(graph), end - start
            )
        )
        # Reset all friend trees to close the files
        for ch in [*self.tchains, *self.friend_tchains]:
            ch.Reset()
        return results

    def run_locally(self, output, nworkers=1, nthreads=1):
        """Save to file the histograms booked.

        Args:
            output (str): Name of the output .root file
            nworkers (int): number of slaves passed to the
                multiprocessing.Pool() function
            nthreads (int): number of threads passed to the
                EnableImplicitMT function
        """
        if not isinstance(nthreads, int):
            raise TypeError("wrong type for nthreads")
        if nthreads < 1:
            raise ValueError("nthreads has to be larger zero")
        self.nthreads = nthreads
        if not isinstance(nworkers, int):
            raise TypeError("wrong type for nworkers")
        if nworkers < 1:
            raise ValueError("nworkers has to be larger zero")
        logger.info(
            "Start computing locally results of {} graphs using {} workers with {} thread(s) each".format(
                len(self.graphs), nworkers, nthreads
            )
        )
        start = time.time()
        if nworkers == 1:
            final_results = list(map(self._run_multiprocess, self.graphs))
        else:
            with Pool(nworkers) as pool:
                final_results = list(pool.map(self._run_multiprocess, self.graphs))
        final_results = [j for i in final_results for j in i]
        end = time.time()
        logger.info("Finished computations in {} seconds".format(int(end - start)))
        logger.info(
            "Write {} results from {} graphs to file {}".format(
                len(final_results), len(self.graphs), output
            )
        )
        root_file = TFile(output, "RECREATE")
        for op in final_results:
            op.Write()
        root_file.Close()

    def node_to_root(self, node, final_results=None, rcw=None, primary_nodes=None):
        if final_results is None:
            final_results = list()
        if primary_nodes is None:
            primary_nodes = list()
        if node.kind == "dataset":
            logger.debug(
                "%%%%%%%%%% node_to_root, converting to ROOT language the following dataset node\n{}".format(
                    node
                )
            )
            result = self.__rdf_from_dataset(node.unit_block)
            prim_node = result.frame
            if prim_node not in primary_nodes:
                primary_nodes.append(prim_node)
        elif node.kind == "selection":
            if len(node.children) > 1:
                logger.debug(
                    "%%%%%%%%%% node_to_root, converting to ROOT language the following crossroad node\n{}".format(
                        node
                    )
                )
            result = self.__cuts_and_weights_from_selection(rcw, node.unit_block)
        elif node.kind == "action":
            logger.debug(
                "%%%%%%%%%% node_to_root, converting to ROOT language the following action node\n{}".format(
                    node
                )
            )
            if isinstance(node.unit_block, Count):
                result = self.__sum_from_count(rcw, node.unit_block)
            elif isinstance(node.unit_block, Histogram):
                result = self.__histo1d_from_histo(rcw, node.unit_block)
        if node.children:
            for child in node.children:
                self.node_to_root(child, final_results, result, primary_nodes)
        else:
            final_results.append(result)
        return final_results, primary_nodes

    def __rdf_from_dataset(self, dataset):
        t_names = [ntuple.directory for ntuple in dataset.ntuples]
        if len(set(t_names)) == 1:
            tree_name = t_names.pop()
        else:
            raise NameError("Impossible to create RDataFrame with different tree names")
        chain = TChain()
        ftag_fchain = {}
        for ntuple in dataset.ntuples:
            chain.Add("{}/{}".format(ntuple.path, ntuple.directory))
            for friend in ntuple.friends:
                if friend.tag not in ftag_fchain.keys():
                    ftag_fchain[friend.tag] = TChain()
                ftag_fchain[friend.tag].Add(
                    "{}/{}".format(friend.path, friend.directory)
                )
        for tag, ch in ftag_fchain.items():
            if tag is None:
                chain.AddFriend(ch)
            else:
                chain.AddFriend(ch, tag)
            # Keep friend chains alive
            self.friend_tchains.append(ch)
        if self.nthreads != 1:
            EnableImplicitMT(self.nthreads)
        # Keep main chain alive
        self.tchains.append(chain)
        rdf = RDataFrame(chain)
        rcw = RDataFrameCutWeight(rdf)
        return rcw

    def __cuts_and_weights_from_selection(self, rcw, selection):
        l_cuts = [cut for cut in rcw.cuts]
        l_weights = [weight for weight in rcw.weights]
        for cut in selection.cuts:
            l_cuts.append(cut)
        for weight in selection.weights:
            l_weights.append(weight)
        l_rcw = RDataFrameCutWeight(rcw.frame, l_cuts, l_weights)
        return l_rcw

    def __sum_from_count(self, rdf, count):
        return rdf.Sum(count.variable)

    def __histo1d_from_histo(self, rcw, histogram):
        name = histogram.name
        var = histogram.variable
        edges = histogram.edges
        nbins = len(edges) - 1

        # Create macro weight string from sub-weights applied
        # (saved earlier as rdf columns)
        weight_expression = "*".join(["(" + weight.expression + ")" for weight in rcw.weights])
        weight_expression = weight_expression.replace("\n", "").replace(" ", "")

        # Create macro cut string from sub-cuts applied
        # (saved earlier as rdf columns)
        cut_name = name.replace("#", "_")
        cut_name = cut_name.replace("-", "_")
        cut_name = "cut_" + cut_name
        cut_expression = " && ".join(["(" + cut.expression + ")" for cut in rcw.cuts])
        cut_expression = cut_expression.replace("\n", "").replace(" ", "")
        if cut_expression:
            if logger.getEffectiveLevel() == logging.DEBUG:
                for cut in rcw.cuts:
                    rcw.frame = rcw.frame.Filter(
                        cut.expression, cut_name + ":" + cut.name
                    )
            else:
                rcw.frame = rcw.frame.Filter(cut_expression)
            # Check for assignments in cut expression
            if re.search("(?<!(=|!|<|>))=(?!=)", cut_expression) is not None:
                logger.warning("Found assignment in cut string. Is this intended?")

        # Create std::vector with the histogram edges
        l_edges = vector["double"]()
        for edge in edges:
            l_edges.push_back(edge)

        if not weight_expression and self.create_histograms:
            # If the histogram variable is built from different columns,
            # define a column with the expression first and fill this
            # new column in the histogram.
            if re.search("(&&|\|\||\+|-|\*|/|<=|>=|<|>|==|!=)", var):
                varname = name.split("#")[-1]
                rcw.frame = rcw.frame.Define(varname, var)
                logger.debug("%%%%%%%%%% Attaching histogram called {}".format(name))
                histo = rcw.frame.Histo1D((name, name, nbins, l_edges.data()), varname)
            else:
                logger.debug("%%%%%%%%%% Attaching histogram called {}".format(name))
                histo = rcw.frame.Histo1D((name, name, nbins, l_edges.data()), var)
        else:
            weight_name = name.replace("#", "_")
            weight_name = weight_name.replace("-", "_")
            if self.create_histograms:
                rcw.frame = rcw.frame.Define(weight_name, weight_expression)
                logger.debug("%%%%%%%%%% Attaching histogram called {}".format(name))
                # If the histogram variable is built from different columns,
                # define a column with the expression first and fill this
                # new column in the histogram.
                if re.search("(&&|\|\||\+|-|\*|/|<=|>=|<|>|==|!=)", var):
                    varname = name.split("#")[-1]
                    rcw.frame = rcw.frame.Define(varname, var)
                    histo = rcw.frame.Histo1D(
                        (name, name, nbins, l_edges.data()), varname, weight_name
                    )
                else:
                    histo = rcw.frame.Histo1D(
                        (name, name, nbins, l_edges.data()), var, weight_name
                    )

        if self.create_config:
            process, channel_and_sample, variation, variable = name.split("#")
            channel, sample = channel_and_sample.split("-")[0], "-".join(channel_and_sample.split("-")[1:])

            variation = variation.replace(f"_{variable}", "").replace("Era", "2018")

            if process == "data":
                sample = "data"
            if not weight_expression:
                weight_expression = "(float)1."
            if not cut_expression:
                cut_expression = "(float)1."
            self.config[channel][process][sample][variation]["cut"] = f"{cut_expression}"
            self.config[channel][process][sample][variation]["weight"] = f"{weight_expression}"

        if self.create_histograms:
            return histo

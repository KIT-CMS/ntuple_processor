from .utils import Count
from .utils import Histogram

from ROOT import RDataFrame
from ROOT import TFile
from ROOT import TChain
from ROOT import EnableImplicitMT
from ROOT.std import vector

import logging
logger = logging.getLogger(__name__)



class RunManager:
    """Convert the Graph-style language into PyROOT/RDataFrame
    language and schedule RDataFrame operations, like the
    following:
        Dataset()     -->   RDataFrame()
        Selection()   -->   Filter()
        Count()   -->   Sum()
        Histogram()   -->   Histo1D()

    Args:
        graphs (list): List of Graph objects that are converted
            node by node to RDataFrame operations

    Attributes:
        final_ptrs (list): List of TH1D objects resulting from a
            set of Filter operations performed on RDataFrames; on
            all them we need to perform a Write operation
    """
    def __init__(self, graphs,
            parallelize = False,
            nthreads = 0):
        self.final_ptrs = list()
        self.tchains = list()
        self.friend_tchains = list()
        self.parallelize = parallelize
        self.nthreads = nthreads
        for graph in graphs:
            # This gets the name of the graph being used
            # (which is also the name of the dataset
            # related to this graph), to put it in the
            # histogram name.
            self._last_used_dataset = graph.name
            #logger.debug('Last used dataset called {}'.format(
                #self._last_used_dataset))
            self.__node_to_root(graph)
        logger.debug('%%%%%%%%%% Final pointers (histograms and counts): {}'.format(
            self.final_ptrs))

    def run_locally(self, of_name, update = False):
        """Save to file the histograms booked.

        Args:
            of_name (str): Name of the output .root
                file
        """
        logger.debug('%%%%%%%%%% Chains {} and friend chains {} still alive'.format(
            self.tchains, self.friend_tchains))
        if update:
            root_file = TFile(of_name, 'UPDATE')
        else:
            root_file = TFile(of_name, 'RECREATE')
        for op in self.final_ptrs:
            op.Write()
        root_file.Close()

    def __node_to_root(self, node, rdf = None):
        logger.debug('%%%%%%%%%% __node_to_root, converting from Graph to ROOT language the following node\n{}'.format(
            node))
        if node.kind == 'dataset':
            result = self.__rdf_from_dataset(
                node.unit_block)
        elif node.kind == 'selection':
            result = self.__cuts_and_weights_from_selection(
                rdf, node.unit_block)
        elif node.kind == 'action':
            if isinstance(node.unit_block, Count):
                result = self.__sum_from_count(
                    rdf, node.unit_block)
            elif isinstance(node.unit_block, Histogram):
                result = self.__histo1d_from_histo(
                    rdf, node.unit_block, self._last_used_dataset)
        if node.children:
            for child in node.children:
                logger.debug('%%%%% __node_to_root, do not return; apply actions in "{}" on RDF "{}"'.format(
                    child.__repr__(), result))
                self.__node_to_root(child, result)
        else:
            logger.debug('%%%%% __node_to_root, final return: append \n{} to final pointers'.format(
                result))
            self.final_ptrs.append(result)

    def __rdf_from_dataset(self, dataset):
        t_names = [ntuple.directory for ntuple in \
            dataset.ntuples]
        if len(set(t_names)) == 1:
            tree_name = t_names.pop()
        else:
            raise NameError(
                'Impossible to create RDataFrame with different tree names')
        chain = TChain(tree_name, tree_name)
        ftag_fchain = {}
        for ntuple in dataset.ntuples:
            logger.debug('%%%%% Dataset -> RDF, processing ntuple {}'.format(
                ntuple))
            chain.Add(ntuple.path)
            for friend in ntuple.friends:
                logger.debug('%%%%% Dataset -> RDF, processing friend {}'.format(
                    friend))
                if friend.tag not in ftag_fchain.keys():
                    ftag_fchain[friend.tag] = TChain(friend.directory, friend.directory)
                    logger.debug('%%%%% Dataset -> RDF, chain created from friend')
                ftag_fchain[friend.tag].Add(friend.path)
        logger.debug('%%%%% Dataset -> RDF, Tags-Chains dictionary: {}'.format(
            ftag_fchain))
        for ch in ftag_fchain.values():
            chain.AddFriend(ch)
            # Keep friend chains alive
            self.friend_tchains.append(ch)
        logger.debug('%%%%% Creating RDF from TChain ({}) with friends {}'.format(
            chain, [f for f in chain.GetListOfFriends()]))
        if self.parallelize:
            EnableImplicitMT(self.nthreads)
        # Keep main chain alive
        self.tchains.append(chain)
        rdf = RDataFrame(chain)
        return rdf

    def __cuts_and_weights_from_selection(self, rdf, selection):
        if selection.cuts:
            cut_name = '__cut__' + selection.name
            cut_expression = ' && '.join([cut.expression for cut in selection.cuts])
            logger.debug('%%%%% Definig merged cut {} column'.format(
                cut_expression))
            l_rdf = rdf.Define(
                cut_name,
                cut_expression)
            rdf = l_rdf
        if selection.weights:
            weight_name = '__weight__' + selection.name
            weight_expression = '*'.join([
                weight.expression for weight in selection.weights])
            l_rdf = rdf.Define(
                weight_name,
                weight_expression)
            logger.debug('%%%%% Defining {} column with weight expression {}'.format(
                weight_name,
                weight_expression))
            rdf = l_rdf
        return rdf

    def __sum_from_count(self, rdf, count):
        return rdf.Sum(count.variable)

    def __histo1d_from_histo(self, rdf, histogram, dataset_name):
        name = histogram.name
        nbins = histogram.binning.nbins
        edges = histogram.binning.edges
        var = histogram.variable

        # Create macro weight string from sub-weights applied
        # (saved earlier as rdf columns)
        weight_expression = '*'.join([
            name for name in rdf.GetColumnNames() if name.startswith(
                '__weight__')])
        logger.debug('%%%%%%%%%% Histo1D from histo: created weight expression {}'.format(
            weight_expression))

        # Create macro cut string from sub-cuts applied
        # (saved earlier as rdf columns)
        cut_expression = ' && '.join([
            name for name in rdf.GetColumnNames() if name.startswith(
                '__cut__')])
        logger.debug('%%%%%%%%%% Histo1D from histogram: created cut expression {}'.format(
            cut_expression))
        if cut_expression:
            l_rdf = rdf.Filter(cut_expression)
            rdf = l_rdf

        # Create std::vector with the histogram edges
        l_edges = vector['double']()
        for edge in edges:
            l_edges.push_back(edge)

        if not weight_expression:
            histo = rdf.Histo1D((
                    name, name, nbins, l_edges.data()),
                    var)
        else:
            weight_name = 'Weight'
            logger.debug('%%%%%%%%%% Histo1D from histogram: defining {} column with weight expression {}'.format(
                weight_name, weight_expression))
            l_rdf = rdf.Define(weight_name, weight_expression)
            histo = l_rdf.Histo1D((
                name, name, nbins, l_edges.data()),
                var, weight_expression)

        return histo
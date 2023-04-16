def preprocess(attributes) -> {}:
    from caddo_tool.defaults.preprocess.rewrite_preprocess import RewritePreprocess
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    return RewritePreprocess().run(attributes, ['y__class'])


def init_model(attributes) -> {}:
    from sklearn.tree import DecisionTreeClassifier
    from caddo_tool.modules.attributes import Attributes
    attributes[Attributes.MODEL] = DecisionTreeClassifier()
    return attributes


def train(attributes) -> {}:
    from caddo_tool.defaults.train.simple_train import SimpleTrain
    """
    Make some training routines on provided data
    :param attributes: data loaded and preprocessed above
    :return: ready to test network
    """
    return SimpleTrain().run(attributes)


def test(attributes) -> {}:
    from caddo_tool.defaults.test.simple_test import SimpleTest
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    return SimpleTest().run(attributes)


def evaluate(attributes):
    from caddo_tool.defaults.evaluate.evaluate_all import EvaluateAll
    return EvaluateAll().run(attributes)


def summarize(attributes):
    from caddo_tool.defaults.summarize.summarize_all import SummarizeAll
    SummarizeAll().run(attributes, './figures')


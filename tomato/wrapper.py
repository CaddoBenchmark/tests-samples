def preprocess(attributes) -> {}:
    from caddo_tool.modules.attributes import Attributes
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    attributes[Attributes.X] = attributes[Attributes.X_RAW].apply(lambda x: [1 for _ in x])
    attributes[Attributes.Y] = attributes[Attributes.Y_RAW]
    return attributes


def init_model(attributes) -> {}:
    from sklearn.tree import DecisionTreeClassifier
    from caddo_tool.modules.attributes import Attributes
    attributes[Attributes.MODEL] = DecisionTreeClassifier()
    return attributes


def train(attributes) -> {}:
    from caddo_tool.modules.attributes import Attributes
    """
    Make some training routines on provided data
    :param attributes: data loaded and preprocessed above
    :return: ready to test network
    """
    attributes[Attributes.MODEL].fit(attributes[Attributes.X], attributes[Attributes.Y])
    return attributes


def test(attributes) -> {}:
    from caddo_tool.modules.attributes import Attributes
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    attributes[Attributes.STORE]['funny1'] = 0
    attributes[Attributes.Y] = attributes[Attributes.MODEL].predict(attributes[Attributes.X])
    return attributes


def evaluate(attributes):
    from caddo_tool.modules.attributes import Attributes
    from sklearn.metrics import accuracy_score
    import random
    acc = accuracy_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y]) - random.randint(3, 9) / 50
    if 'acc' not in attributes[Attributes.STORE]:
        attributes[Attributes.STORE]['acc'] = [acc]
    else:
        attributes[Attributes.STORE]['acc'].append(acc)
    return attributes


def summarize(attributes):
    from caddo_tool.modules.attributes import Attributes
    import matplotlib.pyplot as plt
    acc = attributes[Attributes.STORE]['acc']
    runs = [x for x in range(len(acc))]
    print(acc)
    plt.plot(runs, acc)
    plt.savefig('fig.png')


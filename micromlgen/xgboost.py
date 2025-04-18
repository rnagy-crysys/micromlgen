from micromlgen.utils import jinja, check_type
from tempfile import NamedTemporaryFile
import json


def format_tree(tree):
    """
    Format xgboost tree like a sklearn DecisionTree
    :param tree:
    :return:
    """
    split_indices = tree['split_indices']
    split_conditions = tree['split_conditions']
    left_children = tree['left_children']
    right_children = tree['right_children']
    return {
        'left': left_children,
        'right': right_children,
        'features': split_indices,
        'thresholds': split_conditions
    }


def is_xgboost(clf):
    """
    Test if classifier can be ported
    """
    return check_type(clf, 'XGBClassifier')


def port_xgboost(clf, tmp_file=None, **kwargs):
    """
    Port a XGBoost classifier
    @updated 1.1.28
    :param clf:
    :param tmp_file: if not None, use the given file as temporary export destination
    """
    if tmp_file is None:
        with NamedTemporaryFile('w+', suffix='.json', encoding='utf-8') as tmp:
            clf.save_model(tmp.name)
            tmp.seek(0)
            decoded = json.load(tmp)
    else:
        clf.save_model(tmp_file)

        with open(tmp_file, encoding='utf-8') as file:
            decoded = json.load(file)

    trees = [format_tree(tree) for tree in decoded['learner']['gradient_booster']['model']['trees']]

    return jinja('xgboost/xgboost.jinja', {
        'n_classes': int(decoded['learner']['learner_model_param']['num_class']),
        'trees': trees,
    }, {
        'classname': 'XGBClassifier'
    }, **kwargs), jinja('_skeleton_h.jinja', {'classname': 'XGBClassifier'})
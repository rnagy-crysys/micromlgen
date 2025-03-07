from micromlgen.utils import jinja, check_type


def is_adaboost(clf):
    """Test if classifier can be ported"""
    return check_type(clf, 'AdaBoostClassifier')


def port_adaboost(clf, **kwargs):
    """Port sklearn's AdaBoostClassifier"""
    return jinja('adaboost/adaboost.jinja', {
        'n_classes': clf.n_classes_,
        'weights': clf.estimator_weights_,
        'trees': [{
            'left': clf.tree_.children_left,
            'right': clf.tree_.children_right,
            'features': clf.tree_.feature,
            'thresholds': clf.tree_.threshold,
            'classes': clf.tree_.value,
        } for clf in clf.estimators_]
    }, {
        'classname': 'AdaBoost'
    }, **kwargs), jinja('_skeleton_h.jinja', {'classname': 'AdaBoost'})
# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import pytest

from art.attacks.poisoning import GradientMatchingAttack
from art.utils import to_categorical

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("tensorflow2")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
        classifier, _ = image_dl_estimator()

        class_source = 0
        class_target = 1
        index_target = np.where(y_test.argmax(axis=1) == class_source)[0][5]
        x_trigger = x_test[index_target : index_target + 1]

        x_train, y_train = x_train[:1000], y_train[:1000]
        y_train = np.argmax(y_train, axis=-1)
        attack = GradientMatchingAttack(
            classifier, epsilon=0.3, percent_poison=0.01, max_trials=1, max_epochs=1, verbose=False
        )
        x_poison, y_poison = attack.poison(x_trigger, [class_target], x_train, y_train)

        np.testing.assert_equal(np.shape(x_poison), np.shape(x_train))
        np.testing.assert_equal(np.shape(y_poison), np.shape(y_train))
    except ARTTestException as e:
        art_warning(e)

# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
"""
This module implements the PrincipleUAP attack.
This attack was originally implemented by Khrulkov and Oseledets (2018) as SingularFool.
However, this implementation modifies to conduct the attack with p=q=2 and targeted attacks.

| Paper link: https://arxiv.org/abs/1709.03582
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import scipy.linalg
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class PrincipleUAP(EvasionAttack):
    """
    This attack was originally implemented by Khrulkov and Oseledets (2018) as SingularFool.
    However, this implementation modifies to conduct the attack with p=q=2 and targeted attacks.

    | Paper link: https://arxiv.org/abs/1709.03582
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "targeted",
        "batch_size",
        "summary_writer",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        targeted: bool = False,
        max_iter: int = 5,
        batch_size: int = 32,
        summary_writer: Union[str, bool, SummaryWriter] = False,
    ) -> None:
        """
        Create a :class:`PrincipleUAP` instance.

        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf or 2.
        :param eps: Attack step size (input variation).
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        """
        super().__init__(estimator=estimator, summary_writer=summary_writer)
        self.norm = norm
        self.eps = eps
        self._targeted = targeted
        self.batch_size = batch_size
        PrincipleUAP._check_params(self)

        self._batch_id = 0
        self._i_max_iter = 0

    def _check_compatibility_input_and_eps(self, x: np.ndarray):
        """
        Check the compatibility of the input with `eps` and `eps_step` which are of the same shape.

        :param x: An array with the original inputs.
        """
        if isinstance(self.eps, np.ndarray):
            # Ensure the eps array is broadcastable
            if self.eps.ndim > x.ndim:  # pragma: no cover
                raise ValueError("The `eps` shape must be broadcastable to input shape.")


    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        if isinstance(self.estimator, ClassifierMixin):
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:  # pragma: no cover
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for Principle UAP.")
                y_array = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore
            else:
                y_array = y

            if self.estimator.nb_classes > 2:
                y_array = y_array / np.sum(y_array, axis=1, keepdims=True)
        
        else:
            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:  # pragma: no cover
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for Principle UAP.")
                y_array = self.estimator.predict(x, batch_size=self.batch_size)
            else:
                y_array = y
        
        # get data jacobian
        jacobian = self._compute(x, y_array)
        jacobian = jacobian.reshape(len(x), np.prod(x[0].shape))
        # principle component from data jacobian
        w, v = scipy.linalg.eigh(np.dot(jacobian.T, jacobian) / len(x), check_finite=True)
        logger.info("Variance explained: %.2f%%", 100 * np.max(w.real) / np.sum(w.real))
        noise = v[:, np.argmax(w.real)].real.reshape(x[0].shape)
        del(v)
        del(w)

        if self.norm in [np.inf, "inf"]:
            noise = np.sign(noise)

        noise = self.eps * noise
        
        # check optimal direction
        tmp_x_adv = x + noise
        sr_p = 100 * compute_success(
                self.estimator,  # type: ignore
                x,
                y_array,
                tmp_x_adv,
                self.targeted,
                batch_size=self.batch_size,
            )
        
        tmp_x_adv = x - noise
        sr_n = 100 * compute_success(
                self.estimator,  # type: ignore
                x,
                y_array,
                tmp_x_adv,
                self.targeted,
                batch_size=self.batch_size,
            )

        if sr_p > sr_n:
            sr = sr_p
            x_adv = x + noise
        else:
            sr = sr_n
            x_adv = x - noise

        logger.info("Success rate of Principle UAP: %.2f%%", sr)

        if self.summary_writer is not None:
            self.summary_writer.reset()

        self.noise = x_adv[0] - x[0]

        return x_adv

    def _check_params(self) -> None:

        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')

        if not (
            isinstance(self.eps, (int, float))
            or isinstance(self.eps, np.ndarray)
        ):
            raise TypeError(
                "The perturbation size `eps` must have the same type of `int`"
                ", `float`, or `np.ndarray`."
            )

        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError("The perturbation size `eps` has to be nonnegative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be nonnegative.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")


    def _compute_jacobian(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=grad,
                patch=None,
                estimator=self.estimator,
                x=x,
                y=y,
                targeted=self.targeted,
            )

        # Check for NaN before normalisation an replace with 0
        if grad.dtype != object and np.isnan(grad).any():  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = np.where(np.isnan(grad), 0.0, grad)
        else:
            for i, _ in enumerate(grad):
                grad_i_array = grad[i].astype(np.float32)
                if np.isnan(grad_i_array).any():
                    grad[i] = np.where(np.isnan(grad_i_array), 0.0, grad_i_array).astype(object)

        if (grad.dtype != object and np.isinf(grad).any()) or np.isnan(  # pragma: no cover
            grad.astype(np.float32)
        ).any():
            logger.info("The loss gradient array contains at least one positive or negative infinity.")

        assert x.shape == grad.shape

        return grad


    def _compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_id_ext: Optional[int] = None,
    ) -> np.ndarray:
        
        jacobian = np.zeros((x.shape)).astype(ART_NUMPY_DTYPE)

        # Compute jacobian with implicit batching
        for batch_id in tqdm(range(int(np.ceil(x.shape[0] / float(self.batch_size))))):
            if batch_id_ext is None:
                self._batch_id = batch_id
            else:
                self._batch_id = batch_id_ext
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get jacobian for the batch
            jacobian_batch = self._compute_jacobian(batch, batch_labels)
            # bind
            jacobian[batch_index_1:batch_index_2] = jacobian_batch

        return jacobian


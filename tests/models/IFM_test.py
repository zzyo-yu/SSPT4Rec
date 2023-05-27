# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import IFM
from ..utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((32,), 3),
     ((32,), 2), ((32,), 1),
     ]
)
def test_IFM(hidden_size, sparse_feature_num):
    model_name = "IFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=sparse_feature_num)

    model = IFM(feature_columns, feature_columns,
                dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass

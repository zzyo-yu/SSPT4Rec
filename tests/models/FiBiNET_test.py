# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import FiBiNET
from ..utils import check_model, SAMPLE_SIZE, get_test_data, get_device


@pytest.mark.parametrize(
    'bilinear_type',
    ["each",
     "interaction", "all"]
)
def test_FiBiNET(bilinear_type):
    model_name = "FiBiNET"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=3, dense_feature_num=3)

    model = FiBiNET(feature_columns, feature_columns,
                    bilinear_type=bilinear_type, dnn_hidden_units=[8, 8], dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass

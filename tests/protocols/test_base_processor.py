"""Array-type normalization contract for the base logits processor
(torch/numpy/list/tuple; mlx/jax via guarded imports — DESIGN.md SS4.3/SS12)."""

import numpy as np
import pytest
import torch

from grid.processors import GridBaseLogitsProcessor

arrays = {
    "list": [[1.0, 2.0], [3.0, 4.0]],
    "tuple": ((1.0, 2.0), (3.0, 4.0)),
    "np": np.array([[1, 2], [3, 4]], dtype=np.float32),
    "torch": torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
}


class MockLogitsProcessor(GridBaseLogitsProcessor):
    def process_logits(self, input_ids, logits):
        return logits * 2


@pytest.fixture
def processor():
    return MockLogitsProcessor()


@pytest.mark.parametrize("array_type", arrays.keys())
def test_roundtrip(array_type, processor):
    data = arrays[array_type]
    ids = arrays[array_type]
    out = processor(ids, data)
    assert type(out) is type(data)
    as_t = processor._to_torch(out)
    assert torch.equal(as_t.float(), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))


def test_1d_is_unsqueezed(processor):
    out = processor([1.0, 2.0], [1.0, 2.0])
    assert out == [2.0, 4.0]


def test_unsupported_type_raises(processor):
    with pytest.raises(TypeError):
        processor("nope", "nope")

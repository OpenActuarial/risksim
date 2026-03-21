import numpy as np

from risksim.protocols import SupportsMoments, SupportsSample


class SampleOnlyModel:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.zeros(size, dtype=float)


class SampleAndMomentsModel:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.ones(size, dtype=float)

    def mean(self) -> float:
        return 1.0

    def variance(self) -> float:
        return 0.0


def test_supports_sample_runtime_check() -> None:
    model = SampleOnlyModel()
    assert isinstance(model, SupportsSample)
    assert not isinstance(model, SupportsMoments)


def test_supports_moments_runtime_check() -> None:
    model = SampleAndMomentsModel()
    assert isinstance(model, SupportsSample)
    assert isinstance(model, SupportsMoments)
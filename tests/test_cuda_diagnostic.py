from types import SimpleNamespace

from tools.diagnose_cuda import probe_torch


def fake_torch(cuda):
    return SimpleNamespace(__version__="2.0.1+cu118", __file__="fake/torch.py",
                           version=SimpleNamespace(cuda="11.8"), cuda=cuda)


def test_diagnostic_preserves_real_init_error_when_gpu_unavailable():
    def fail_init():
        raise RuntimeError("driver initialization failed")

    cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0, init=fail_init)
    report = probe_torch(fake_torch(cuda))
    assert report["ok"] is False
    assert report["cuda_available"] is False
    assert report["device_count"] == 0
    assert report["error"] == "RuntimeError: driver initialization failed"


def test_diagnostic_requires_actual_scalar_compute():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           init=lambda: None, get_device_name=lambda _: "fake GPU",
                           get_device_capability=lambda _: (8, 9), synchronize=lambda: None)
    torch = fake_torch(cuda)

    class Scalar:
        def __add__(self, other):
            assert other == 1
            return self

        def item(self):
            return 2.0

    torch.ones = lambda size, device: Scalar()
    report = probe_torch(torch)
    assert report["ok"] is True
    assert report["scalar_probe"] == 2.0


def test_diagnostic_does_not_treat_visibility_as_compute_success():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           init=lambda: None, get_device_name=lambda _: "fake GPU",
                           get_device_capability=lambda _: (8, 9))
    report = probe_torch(fake_torch(cuda))
    assert report["ok"] is False
    assert "error" in report

import pytest

from bench import stream

# Captured from a real run of the pinned STREAM build (2 threads, cloud VM).
SAMPLE = """\
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
This system uses 8 bytes per array element.
-------------------------------------------------------------
Array size = 50000000 (elements), Offset = 0 (elements)
Number of Threads requested = 2
Number of Threads counted = 2
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           37789.5     0.039652     0.021170     0.104243
Scale:          23109.9     0.059919     0.034617     0.116648
Add:            27443.6     0.075150     0.043726     0.104035
Triad:          27655.5     0.074385     0.043391     0.109798
-------------------------------------------------------------
Solution Validates: avg error less than 1.000000e-13 on all three arrays
-------------------------------------------------------------
"""


def test_parse_sample():
    r = stream.parse(SAMPLE)
    assert r["array_elements"] == 50_000_000
    assert r["threads_counted"] == 2
    assert r["validates"] is True
    assert r["kernels"]["Triad"]["best_rate_gbs"] == pytest.approx(27.6555)
    assert r["kernels"]["Copy"]["min_time_s"] == pytest.approx(0.021170)


def test_parse_rejects_truncated_output():
    with pytest.raises(stream.StreamError):
        stream.parse(SAMPLE.split("Add:")[0])


def test_best_by_threads():
    rows = [{"threads": 1, "kernel": "Triad", "best_rate_gbs": 10.0},
            {"threads": 1, "kernel": "Triad", "best_rate_gbs": 12.0},
            {"threads": 4, "kernel": "Triad", "best_rate_gbs": 30.0},
            {"threads": 4, "kernel": "Copy", "best_rate_gbs": 99.0}]
    assert stream.best_by_threads(rows) == {1: 12.0, 4: 30.0}


def test_default_threads_includes_endpoints():
    t = stream.default_threads()
    assert t[0] == 1 and t == sorted(set(t))

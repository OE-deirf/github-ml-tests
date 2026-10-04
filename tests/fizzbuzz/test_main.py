from runpy import run_module
from pytest import CaptureFixture

from src.fizzbuzz import fizzbuzz_range, LIMIT


def test_python_m_fizzbuzz(capsys: CaptureFixture[str]):
    run_module("src.fizzbuzz", run_name="__main__")
    assert capsys.readouterr().out.split() == fizzbuzz_range(LIMIT)

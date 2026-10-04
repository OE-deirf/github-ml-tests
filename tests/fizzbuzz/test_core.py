""" fizzbuzz test cases """

import pytest
from src.fizzbuzz import fizzbuzz, fizzbuzz_range, LIMIT


@pytest.mark.parametrize("n,expected", [
    (1, "1"), (2, "2"), (3, "Fizz"), (5, "Buzz"),
    (9, "Fizz"), (10, "Buzz"), (15, "FizzBuzz"),
    (30, "FizzBuzz"), (98, "98"),
])
def test_one_value(n: int, expected: str):
    """One value case"""
    assert fizzbuzz(n) == expected


def test_length():
    """test limit"""
    assert len(fizzbuzz_range(LIMIT)) == LIMIT


def test_increase_the_limit():
    """Range of fizbuzz follow the limit"""
    assert fizzbuzz_range(LIMIT)[-1] == fizzbuzz(LIMIT)


def test_fizzbuzz_count():
    """Count of FizzBuzz text durig the limit"""
    eredmeny = fizzbuzz_range(LIMIT)
    assert eredmeny.count("FizzBuzz") == LIMIT // 15

import pytest

from blooms import LEVELS, get_verbs, normalize_level, sample_verbs, validate_question, verb_level


def test_all_levels_and_unique_verbs():
    allv = [v for l in LEVELS for v in get_verbs(l)]
    assert len(allv) == len(set(allv))
    assert all(len(get_verbs(l)) >= 8 for l in LEVELS)


def test_normalize():
    assert normalize_level(3) == "L3" and normalize_level("bt4") == "L4" and normalize_level("Create") == "L6"
    with pytest.raises(ValueError):
        normalize_level("L9")


def test_verb_level_and_sampling():
    assert verb_level("Justify") == "L5"
    assert len(sample_verbs("L2", 4)) == 4


def test_validator():
    assert validate_question("Define overfitting in machine learning models?", "L1")[0]
    assert not validate_question("Explain overfitting in machine learning models?", "L1")[0]
    assert not validate_question("Define overfitting? Define bias?", "L1")[0]
    assert not validate_question("", "L1")[0]
    assert validate_question("1. **Compare** supervised and unsupervised learning methods.", "L4")[0]

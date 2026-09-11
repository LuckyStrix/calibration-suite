from calsuite import provenance as prov


def test_ordering_strongest_to_weakest():
    assert prov.PROVENANCE == ("measured", "derived", "vendor", "nominal")


def test_rank_orders_measured_strongest():
    assert prov.rank("measured") < prov.rank("derived") < prov.rank("vendor") < prov.rank("nominal")


def test_is_exportable():
    assert prov.is_exportable("measured")
    assert prov.is_exportable("derived")
    assert not prov.is_exportable("vendor")
    assert not prov.is_exportable("nominal")


def test_stronger_or_equal():
    assert prov.stronger_or_equal("measured", "nominal")
    assert prov.stronger_or_equal("measured", "measured")
    assert not prov.stronger_or_equal("nominal", "measured")


def test_rank_raises_on_unknown():
    import pytest

    with pytest.raises(ValueError):
        prov.rank("guessed")

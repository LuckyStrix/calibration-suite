from calsuite.fit import Analysis, Refusal


def test_analysis_ok_with_no_refusals():
    a = Analysis(result={"gain": 2.0})
    assert a.ok


def test_analysis_not_ok_after_refuse():
    a = Analysis(result={})
    a.refuse("coverage", "outer 20% of field has holes", value=0.6, threshold=0.8)
    assert not a.ok
    assert len(a.refusals) == 1
    assert isinstance(a.refusals[0], Refusal)


def test_refusal_to_dict_is_json_safe():
    r = Refusal("clipping", "channel saturated", value=1.0, threshold=0.98)
    d = r.to_dict()
    assert d == {"check": "clipping", "message": "channel saturated", "value": 1.0, "threshold": 0.98}

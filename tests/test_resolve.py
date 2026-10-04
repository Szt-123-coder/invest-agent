import pytest

from app.resolve import Resolved, resolve, to_code
from app.symbols import UnknownSymbol


def fake_model(answer: Resolved):
    calls = []

    def ask(text):
        calls.append(text)
        return answer
    return ask, calls


def test_known_names_and_codes_never_call_the_model():
    ask, calls = fake_model(Resolved(listed=True, symbol="X", name="x"))
    assert resolve("茅台", ask).symbol == "600519.SS"
    assert resolve("shop", ask).symbol == "SHOP"
    assert calls == []


def test_model_resolves_unknown_name_then_cache_is_used():
    ask, calls = fake_model(Resolved(listed=True, symbol="005930.KS", name="三星电子", exchange="韩国交易所"))
    r = resolve("三星电子公司", ask, verify=lambda code: None)
    assert (r.symbol, r.source) == ("005930.KS", "模型")
    assert resolve("三星电子公司", ask).source == "缓存" and len(calls) == 1
    assert to_code("三星电子公司") == "005930.KS"   # agent 的工具也能用上缓存


def test_model_code_that_has_no_data_is_rejected_and_not_cached():
    ask, _ = fake_model(Resolved(listed=True, symbol="FAKE.KS", name="编出来的"))

    def verify(code):
        raise UnknownSymbol("行情接口里没有")
    with pytest.raises(UnknownSymbol, match="查不到"):
        resolve("某公司", ask, verify=verify)
    with pytest.raises(UnknownSymbol):
        resolve("某公司")      # 没进缓存


def test_unlisted_answer_from_model_is_explained():
    ask, _ = fake_model(Resolved(listed=False, name="大疆创新", note="没有上市"))
    with pytest.raises(UnknownSymbol, match="大疆创新：没有上市"):
        resolve("大疆创新科技", ask)
    with pytest.raises(UnknownSymbol, match="没有上市"):
        resolve("华为", ask)

import json

import pytest

from visforge.structured import READ_OUTPUT, OutputContract, OutputError, readable_text


def test_contract_example_is_valid_json_accepted_by_its_own_parser():
    contract = OutputContract({"found": True, "target": "蓝色方块"})
    example = json.loads(contract.instruction[contract.instruction.index("{"):])
    assert contract.parse(json.dumps(example)) == {"found": True, "target": "蓝色方块"}


@pytest.fixture
def contract():
    return OutputContract({"count": 0, "found": True})


def test_only_one_visible_payload_survives_reasoning_and_prose(contract):
    reply = ('<think>{"found":false}</think>Here is the result.\n```json\n'
             '{"count":2,"found":true,"note":"a}b[", "flags":[false]}\n```\nDone.')
    assert contract.parse(reply) == {"count": 2, "found": True, "note": "a}b[", "flags": [False]}
    assert contract.parse('回答如下：{"count":1,"found":true} 完毕') == {"count": 1, "found": True}


def test_quoted_terms_in_surrounding_prose_are_not_answers(contract):
    reply = 'For "the blue [box]", my answer is {"count":2,"found":true}. It was called "item 1".'
    assert contract.parse(reply) == {"count": 2, "found": True}


def test_fenced_reply_can_follow_a_bracketed_prose_note(contract):
    reply = 'Answer [see below]:\r\n```json\r\n{"count":2,"found":true}\r\n```'
    assert contract.parse(reply) == {"count": 2, "found": True}


@pytest.mark.parametrize("reply, subtype", [
    ('{"count":2,"found":"false"}', "wrong_schema"),
    ('{"count":true,"found":true}', "wrong_schema"),
    ('{"count":2,"found":1}', "wrong_schema"),
    ('{"found":false}', "wrong_schema"),
    ('[]', "wrong_schema"),
    ('true', "wrong_schema"),
    ('"[1]"', "wrong_schema"),
    ('setting="ordinary text" other prose', "invalid_json"),
    ('"outer": {"count":1,"found":true}', "invalid_json"),
    ('{"count":1,"found":true}true', "ambiguous_json"),
    ('```json\r\ntrue\r\n```\r\n{"count":1,"found":true}', "ambiguous_json"),
    ('{"nested":{"count":1,"found":true}', "invalid_json"),
    ('answer: [{"count":1,"found":true}', "invalid_json"),
    ('{"count":1,"found":true}]', "invalid_json"),
    ('{"count":1,"found":true} {"count":', "invalid_json"),
    ('{"count":1,"found":true}\n```json\n{}\n```', "ambiguous_json"),
    ('```json\n{}\n```\nActually, {"count":1,"found":true}', "ambiguous_json"),
    ('{"count":1,"found":false,"found":true}', "ambiguous_json"),
    ('{"count":1,"found":true,"note":[{"x":0,"x":1}]}', "ambiguous_json"),
    ('{"count":1,"found":true,"info":[NaN]}', "invalid_json"),
    ('{"count":1,"found":true,"info":-Infinity}', "invalid_json"),
    ('{"count":1,"found":true,"info":1e999}', "invalid_json"),
    ('<think>{"count":1,"found":true}', "invalid_json"),
    ("", "invalid_json"),
    (None, "invalid_json"),
])
def test_bad_reply_is_classified_without_recovering_inner_evidence(contract, reply, subtype):
    with pytest.raises(OutputError) as exc:
        contract.parse(reply)
    assert exc.value.subtype == subtype


@pytest.mark.parametrize("example", [[{"id": 0}], {"readings": [{"id": 0}]}])
def test_collection_envelopes_are_declared_and_unambiguous(example):
    contract = OutputContract(example, list_key="readings")
    assert contract.parse('[{"id":2}]') == [{"id": 2}]
    assert contract.parse('{"readings":[{"id":2}],"unit":"cm"}') == [{"id": 2}]
    assert contract.parse('{"readings":[]}') == []
    for reply in ['{"values":[]}', '{"readings":"[]"}']:
        with pytest.raises(OutputError) as exc:
            contract.parse(reply)
        assert exc.value.subtype == "wrong_schema"
    with pytest.raises(OutputError) as exc:
        contract.parse('{"readings":[],"results":[{"id":1}]}')
    assert exc.value.subtype == "ambiguous_json"


def test_array_envelope_is_disabled_by_default():
    array = OutputContract([{"a": 1}])
    assert array.parse('[{"a":2}]') == [{"a": 2}]
    with pytest.raises(OutputError) as exc:
        array.parse('{"readings":[{"a":2}]}')
    assert exc.value.subtype == "wrong_schema"


def test_read_contract_preserves_transcribed_text_and_explicit_abstention():
    assert readable_text(READ_OUTPUT.parse('{"text":"  TEA  ","legible":true}')) == "  TEA  "
    assert readable_text(READ_OUTPUT.parse('{"legible":false}')) is None
    for reply in ['{"text":1,"legible":true}', '{"text":"  ","legible":true}',
                  '{"legible":true}', '{"legible":"false","text":"TEA"}']:
        with pytest.raises(OutputError) as exc:
            readable_text(READ_OUTPUT.parse(reply))
        assert exc.value.subtype == "wrong_schema"

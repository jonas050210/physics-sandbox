from physics_sandbox.core.components import BUILTIN_TYPES
from physics_sandbox.ui.i18n import SUPPORTED_LANGUAGES, STRINGS, missing_keys, normalize_language, t


def test_german_and_english_have_identical_keys():
    assert missing_keys() == {"missing_in_en": [], "missing_in_de": []}


def test_placeholders_match_between_languages():
    import re
    for key in STRINGS["de"]:
        de = set(re.findall(r"\{(\w+)", STRINGS["de"][key]))
        en = set(re.findall(r"\{(\w+)", STRINGS["en"][key]))
        assert de == en, key


def test_lookup_and_fallback():
    assert t("tool.select", "de") != t("tool.select", "en")
    assert t("does.not.exist", "de") == "does.not.exist"
    assert normalize_language("fr") == "de"
    assert set(SUPPORTED_LANGUAGES) == {"de", "en"}


def test_component_type_names_are_localised():
    for ctype in BUILTIN_TYPES:
        assert ctype.name("de") and ctype.name("en")
        assert ctype.name_de != ctype.name_en


def test_dynamically_built_keys_exist_in_both_languages():
    """Keys assembled at runtime (tool.<x>, cmd.<x>, sim.status.<x>, ...) must exist in DE and EN."""
    from physics_sandbox.ui.i18n import STRINGS
    dynamic = (
        [f"tool.{k}" for k in ("select", "place", "move", "delete")]
        + [f"cmd.{k}" for k in ("add", "remove", "move", "rotate")]
        + [f"sim.status.{k}" for k in ("edit", "running", "paused")]
        + ["sim.start", "sim.pause", "sim.resume", "sim.speed", "status.speed", "hint.sim"]
    )
    for lang in ("de", "en"):
        missing = [k for k in dynamic if k not in STRINGS[lang]]
        assert not missing, (lang, missing)

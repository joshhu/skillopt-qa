from skillopt.optimizer import (
    SEED_SKILL,
    SEED_SKILL_ZH,
    _optimizer_system,
    default_seed_skill,
    is_chinese,
)


def test_is_chinese_matches_zh_prefix():
    assert is_chinese("zh-Hant") is True
    assert is_chinese("zh") is True
    assert is_chinese("en") is False


def test_default_seed_skill_by_language():
    assert default_seed_skill("en") == SEED_SKILL
    assert default_seed_skill("zh-Hant") == SEED_SKILL_ZH
    # 繁中種子確實是中文,英文種子不含中文
    assert "問題" in SEED_SKILL_ZH
    assert "問題" not in SEED_SKILL


def test_optimizer_system_appends_chinese_directive_only_for_zh():
    en = _optimizer_system("en")
    zh = _optimizer_system("zh-Hant")
    assert "Traditional Chinese" not in en      # 英文 pipeline 不受影響
    assert "Traditional Chinese" in zh
    assert zh.startswith(en)                     # 繁中僅在英文骨架後追加規則

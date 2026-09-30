from b01_nuna_lora.benchmark import DEFAULT_TASKS, model_args


def test_default_tasks_are_the_comparable_set():
    assert DEFAULT_TASKS == ("gsm8k", "ifeval", "mmlu")


def test_model_args_name_the_base_and_the_adapter():
    args = model_args(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter="outputs/adapter",
        load_in_4bit=True,
    )
    assert "pretrained=Qwen/Qwen2.5-3B-Instruct" in args
    assert "peft=outputs/adapter" in args
    assert "load_in_4bit=True" in args


def test_model_args_omit_peft_for_a_base_control():
    args = model_args(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter=None,
        load_in_4bit=False,
    )
    assert "peft=" not in args
    assert "load_in_4bit" not in args

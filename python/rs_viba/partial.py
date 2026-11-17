"""`T << X`: partial application (implemented in Rust)."""

from rs_viba._native import (
    answers_the_environment_py as answers_the_environment,
    environment_result_problem_py as environment_result_problem,
    file_environment_result_problem_py as file_environment_result_problem,
    get_args_product_py as get_args_product,
    module_as_function_py as module_as_function,
    names_the_environment_py as names_the_environment,
    parameter_node_py as parameter_node,
    parameters_of_py as parameters_of,
    product_elements_py as product_elements,
    result_of_py as result_of,
)

RET_NAME = "__impl__"
DEF_NAME = "__decl__"
GET_ARGS_NAME = "__get_args__"
ENVIRON_TAG = "$env"
ENV_TYPE = "Env"
ENVIRONMENT_TYPE = "Environment"
DYN_CALL_NAME = "__dyn_call__"
DYN_METHOD_NAME = "__dyn_method__"

__all__ = [
    "reduce_partial", "module_as_function", "product_elements", "parameters_of",
    "parameter_node", "result_of", "get_args_product", "answers_the_environment",
    "names_the_environment", "environment_result_problem",
    "file_environment_result_problem",
]

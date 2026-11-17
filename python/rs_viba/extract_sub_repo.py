"""A sub-repo as one document, with line maps."""

from rs_viba._native import (
    extract_sub_repo_into_viba_markdown_py as _extract_sub_repo_into_viba_markdown,
    extract_sub_viba_markdown_py as _extract_sub_viba_markdown,
)


def extract_sub_repo_into_viba_markdown(repo, viba_path=(),
                                        extract_source_viba_path=(),
                                        type_names=(), never_eqv=None,
                                        nil_eqv=None):
    """Result[(VibaMarkdown, VibaMdToRepoSourceMap)]: the sub-repo as one document.

    `repo` is a directory; `viba_path` the directories an import is looked up
    on, in order; `extract_source_viba_path` the directories of `repo` whose
    definitions are the ones to copy; `type_names` the full names to start from
    (`xxx.yyy.zzz`). `never_eqv` / `nil_eqv` are the unit names, or None for the
    language's own (`reflect.language_config`).

    The answer is the document the sub-repo is, and a map from each of the
    document's code lines to where it came from — the module it is a piece of,
    and the line in that module.
    """
    return _extract_sub_repo_into_viba_markdown(
        repo, viba_path, extract_source_viba_path, type_names,
        never_eqv, nil_eqv)


def extract_sub_viba_markdown(source_viba_md, type_names=(), never_eqv=None,
                              nil_eqv=None):
    """Result[(VibaMarkdown, VibaMdToVibaMdSourceMap)]: a document in, a document out.

    `source_viba_md` is a document's content: one section per file, the section's
    name the file's path in the repo the document stands for. The sub-document
    answers the definitions `type_names` reach, cut the way the repo entry cuts
    them; the map says, for each code line of the answer, the line of the source
    document it came from.
    """
    return _extract_sub_viba_markdown(
        source_viba_md, type_names, never_eqv, nil_eqv)


__all__ = ["extract_sub_repo_into_viba_markdown", "extract_sub_viba_markdown"]

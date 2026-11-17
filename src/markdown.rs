//! markdown: a directory written as one Markdown document (was viba/markdown.py).
//!
//! A directory of files can be given as a single Markdown file — `<dir>.viba.md`,
//! next to `<dir>/`: each heading names one file of the directory, whatever its
//! level. The file's code is every code block under it that is marked `viba`,
//! in the order they stand, glued into one source; the prose between them and
//! any other code block become comments (`# ...`), so the section reads as the
//! file it names and parses as one.

use pyo3::exceptions::PySyntaxError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::pyast;

/// A section's lines: (text, document line) pairs.
pub(crate) type Lines = Vec<(String, usize)>;

/// The files a document names: (name, lines), in the order given.
pub(crate) type Sections = Vec<(String, Lines)>;

/// Why a document does not name its files.
pub(crate) enum ReadError {
    /// A heading with no file name: `##` alone.
    NeedsName(usize),
    /// Two sections name one file.
    DuplicateName(usize, String),
    /// A fenced code block that never closes.
    UnclosedFence,
}

/// Python's `str.splitlines`: a line boundary ends a line, and the last line
/// is not followed by an empty one.
pub(crate) fn split_lines(source: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < source.len() {
        let ch = source[index..].chars().next().unwrap();
        if is_line_boundary(ch) {
            lines.push(&source[start..index]);
            let mut next = index + ch.len_utf8();
            if ch == '\r' && source[next..].starts_with('\n') {
                next += 1;
            }
            start = next;
            index = next;
        } else {
            index += ch.len_utf8();
        }
    }
    if start < source.len() {
        lines.push(&source[start..]);
    }
    lines
}

fn is_line_boundary(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn trim_blanks(lines: Lines) -> Lines {
    let first = lines
        .iter()
        .position(|(text, _at)| !is_blank(text))
        .unwrap_or(lines.len());
    let last = lines
        .iter()
        .rposition(|(text, _at)| !is_blank(text))
        .map(|index| index + 1)
        .unwrap_or(first);
    lines[first..last].to_vec()
}

/// A heading: `#` to `######` then the file's name; `##` alone names nothing.
///
/// The match is Python's `_HEADING.match`, whose second group is the name as
/// written, without the spaces and tabs on either side.
pub(crate) fn match_heading(line: &str) -> Option<(usize, Option<&str>)> {
    let hashes = line.bytes().take_while(|byte| *byte == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &line[hashes..];
    let content = rest.trim_start_matches([' ', '\t']);
    if content.is_empty() {
        if rest.chars().all(|ch| ch == ' ' || ch == '\t') {
            return Some((hashes, None));
        }
        return None;
    }
    if content.len() == rest.len() {
        return None;
    }
    Some((hashes, Some(content.trim_end_matches([' ', '\t']))))
}

/// A fenced code block: up to three spaces of indent, three or more backticks
/// or tildes, and the rest of the line as its info string.
pub(crate) struct Fence<'a> {
    pub indent: usize,
    pub marker: &'a str,
    pub info: &'a str,
}

/// The match of `_FENCE`: the indent, the marker and the info string, each
/// trimmed the way that expression leaves them.
pub(crate) fn match_fence(line: &str) -> Option<Fence<'_>> {
    let bytes = line.as_bytes();
    let mut indent = 0;
    while indent < 3 && indent < bytes.len() && bytes[indent] == b' ' {
        indent += 1;
    }
    let rest = &line[indent..];
    let marker_char = *rest.as_bytes().first()?;
    if marker_char != b'`' && marker_char != b'~' {
        return None;
    }
    let marker_len = rest.bytes().take_while(|byte| *byte == marker_char).count();
    if marker_len < 3 {
        return None;
    }
    let marker = &rest[..marker_len];
    let info = rest[marker_len..].trim_matches([' ', '\t']);
    Some(Fence {
        indent,
        marker,
        info,
    })
}

/// A name written as a code span: its backticks are no part of the name.
fn match_code_span(text: &str) -> Option<&str> {
    let ticks = text.bytes().take_while(|byte| *byte == b'`').count();
    if ticks == 0 {
        return None;
    }
    for width in (1..=ticks).rev() {
        if 2 * width > text.len() {
            continue;
        }
        let end = text.len() - width;
        if text.as_bytes()[end..].iter().all(|byte| *byte == b'`')
            && text.as_bytes()[..width].iter().all(|byte| *byte == b'`')
        {
            return Some(&text[width..end]);
        }
    }
    None
}

/// The file name a heading spells: a code span's content, or the text as written.
fn heading_name(text: &str) -> &str {
    match match_code_span(text) {
        Some(span) => span.trim(),
        None => text,
    }
}

/// One line of a fenced block, the fence's own indent taken off.
pub(crate) fn unindent(line: &str, indent: usize) -> &str {
    let mut boundary = line.len();
    let mut taken = 0;
    for (index, _ch) in line.char_indices() {
        if taken == indent {
            boundary = index;
            break;
        }
        taken += 1;
    }
    if line[..boundary].trim().is_empty() {
        &line[boundary..]
    } else {
        line
    }
}

/// A line that is not the file's code, as a comment in the file.
pub(crate) fn comment(line: &str) -> String {
    if is_blank(line) {
        String::new()
    } else {
        format!("# {line}")
    }
}

/// A file's text: its lines, and a newline after the last one.
pub(crate) fn text(lines: &Lines) -> String {
    let mut out = String::new();
    for (line, _at) in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

struct FenceState {
    indent: usize,
    marker: String,
    lines: Lines,
    viba: bool,
}

/// The files a document names, each line with the document line it came from.
pub(crate) fn read(source: &str) -> Result<Sections, ReadError> {
    let mut files: Sections = Vec::new();
    let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut current: Option<usize> = None;
    let mut fence: Option<FenceState> = None;
    for (offset, line) in split_lines(source).into_iter().enumerate() {
        let number = offset + 1;
        if fence.is_none() {
            if let Some((_level, name)) = match_heading(line) {
                let name = heading_name(name.unwrap_or("")).to_string();
                if name.is_empty() {
                    return Err(ReadError::NeedsName(number));
                }
                if names.contains(&name) {
                    return Err(ReadError::DuplicateName(number, name));
                }
                names.insert(name.clone());
                current = Some(files.len());
                files.push((name, Vec::new()));
                continue;
            }
        }
        let Some(index) = current else {
            continue;
        };
        if let Some(mut state) = fence.take() {
            let closes = match match_fence(line) {
                Some(closed) => {
                    closed.marker.starts_with(state.marker.as_str()) && closed.info.is_empty()
                }
                None => false,
            };
            if closes {
                // The block's lines join the file: a viba block is the code,
                // any other block is documentation and becomes comments.
                let lines = if state.viba {
                    state.lines
                } else {
                    state
                        .lines
                        .into_iter()
                        .map(|(part, at)| (comment(&part), at))
                        .collect()
                };
                files[index].1.extend(lines);
                continue;
            }
            state
                .lines
                .push((unindent(line, state.indent).to_string(), number));
            fence = Some(state);
            continue;
        }
        if let Some(opened) = match_fence(line) {
            let viba = opened.info.split_whitespace().next() == Some("viba");
            fence = Some(FenceState {
                indent: opened.indent,
                marker: opened.marker.to_string(),
                lines: Vec::new(),
                viba,
            });
            continue;
        }
        files[index].1.push((comment(line), number));
    }
    if fence.is_some() {
        return Err(ReadError::UnclosedFence);
    }
    Ok(files
        .into_iter()
        .map(|(name, lines)| (name, trim_blanks(lines)))
        .collect())
}

fn syntax_error(py: Python<'_>, error: ReadError) -> PyErr {
    match error {
        ReadError::NeedsName(number) => {
            PySyntaxError::new_err(format!("line {number}: a section needs a file name"))
        }
        ReadError::DuplicateName(number, name) => {
            let value = PyString::new(py, &name);
            let repr = pyast::repr_of(py, value.as_any());
            PySyntaxError::new_err(format!("line {number}: two sections name the file {repr}"))
        }
        ReadError::UnclosedFence => {
            PySyntaxError::new_err("a fenced code block is not closed at the end of the document")
        }
    }
}

/// The files a Markdown document names: (name, text), in the order given.
#[pyfunction]
fn files_of(py: Python<'_>, source: &str) -> PyResult<Vec<(String, String)>> {
    let files = read(source).map_err(|error| syntax_error(py, error))?;
    Ok(files
        .into_iter()
        .map(|(name, lines)| (name, text(&lines)))
        .collect())
}

/// The files a document names, line by line: (name, [(text, document line)]).
#[pyfunction]
fn lines_of(py: Python<'_>, source: &str) -> PyResult<Sections> {
    read(source).map_err(|error| syntax_error(py, error))
}

/// (file name, module name, text) for the files a document holds.
#[pyfunction]
fn module_files(
    py: Python<'_>,
    module_name: &str,
    source: &str,
) -> PyResult<Vec<(String, String, String)>> {
    let files = read(source).map_err(|error| syntax_error(py, error))?;
    let mut out = Vec::new();
    for (file_name, lines) in files {
        if !file_name.ends_with(".viba") {
            continue;
        }
        let stem = &file_name[..file_name.len() - ".viba".len()];
        let module = format!("{module_name}.{stem}");
        out.push((file_name, module, text(&lines)));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(files_of, m)?)?;
    m.add_function(wrap_pyfunction!(lines_of, m)?)?;
    m.add_function(wrap_pyfunction!(module_files, m)?)?;
    Ok(())
}

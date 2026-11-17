//! repomix: a directory of viba code, packed into one Markdown document
//! (was viba/repomix.py).
//!
//! The text is the one `markdown` reads: a section per file, the section's name
//! is the file's path in the directory, and the code block under it is the
//! file's text. What counts as code: a `.viba` file unless the directory
//! holding it is written as `<dir>.viba.md`, and the sections of a `.viba.md`
//! document, packed under the paths its directory holds. Everything else is
//! left out.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyOSError, PyUnicodeDecodeError};
use pyo3::prelude::*;

use crate::markdown;

/// The code a directory holds: relative path -> (before, code, after).
type Sections = BTreeMap<String, (String, String, String)>;

/// A directory of viba code as one Markdown document's content.
#[pyfunction]
fn viba_repomix(viba_code_dir: &str) -> PyResult<String> {
    let sections = sections_of(Path::new(viba_code_dir))?;
    let mut parts: Vec<String> = Vec::new();
    for (path, (before, code, after)) in &sections {
        let mut lines: Vec<String> = vec![format!("## `{path}`"), String::new()];
        if !before.is_empty() {
            lines.extend(
                markdown::split_lines(before)
                    .into_iter()
                    .map(str::to_string),
            );
        }
        if !code.is_empty() {
            let fence = fence_for(code);
            lines.push(format!("{fence}viba {path}"));
            lines.extend(markdown::split_lines(code).into_iter().map(str::to_string));
            lines.push(fence);
        }
        if !after.is_empty() {
            lines.extend(markdown::split_lines(after).into_iter().map(str::to_string));
        }
        lines.push(String::new());
        parts.push(lines.join("\n"));
    }
    let text = parts.join("\n");
    if text.is_empty() {
        Ok(String::new())
    } else {
        Ok(format!("{}\n", text.trim_end_matches('\n')))
    }
}

/// The code a directory holds: relative path -> (before, code, after).
fn sections_of(viba_code_dir: &Path) -> PyResult<Sections> {
    if !viba_code_dir.is_dir() {
        return Ok(Sections::new());
    }
    let mut found = Sections::new();
    for (relative, path) in collect_files(viba_code_dir) {
        if relative.ends_with(".viba.md") {
            pack_document(&mut found, &relative, &path)?;
            continue;
        }
        if !relative.ends_with(".viba") {
            continue;
        }
        if written_as_document(&path) {
            continue; // the document is the directory: it wins
        }
        let text = file_text(&path)?;
        found.insert(relative, lifted(&text));
    }
    Ok(found)
}

/// A directory written as one document: its sections, under the paths its
/// directory would hold them.
fn pack_document(found: &mut Sections, relative: &str, path: &Path) -> PyResult<()> {
    let directory = &relative[..relative.len() - ".viba.md".len()];
    let text = file_text(path)?;
    let Ok(sections) = markdown::read(&text) else {
        return Ok(()); // a document that does not parse holds no code
    };
    for (file_name, lines) in sections {
        let text = markdown::text(&lines);
        let code = if text.ends_with('\n') || text.is_empty() {
            text
        } else {
            format!("{text}\n")
        };
        found.insert(
            format!("{directory}/{file_name}"),
            (String::new(), code, String::new()),
        );
    }
    Ok(())
}

/// True when this file's directory is written as one document.
fn written_as_document(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let mut document = parent.as_os_str().to_owned();
    document.push(".viba.md");
    Path::new(&document).is_file()
}

/// The files a directory holds, in the order `sorted(root.rglob("*"))` gives
/// them: by path, part by part, keeping the symlinks that point at files.
fn collect_files(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    walk(root, root, &mut found);
    found.sort_by(|left, right| left.0.split('/').cmp(right.0.split('/')));
    found
}

fn walk(root: &Path, directory: &Path, found: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk(root, &path, found);
        } else if file_type.is_symlink() {
            if fs::metadata(&path)
                .map(|metadata| metadata.is_file())
                .unwrap_or(false)
            {
                push_file(root, path, found);
            }
        } else if file_type.is_file() {
            push_file(root, path, found);
        }
    }
}

fn push_file(root: &Path, path: PathBuf, found: &mut Vec<(String, PathBuf)>) {
    if let Some(relative) = relative_path(root, &path) {
        found.push((relative, path));
    }
}

fn relative_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut parts: Vec<String> = Vec::new();
    for component in relative.components() {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    Some(parts.join("/"))
}

/// One file's text, with a newline after the last line.
fn file_text(path: &Path) -> PyResult<String> {
    let text = read_text(path)?;
    Ok(if text.ends_with('\n') {
        text
    } else {
        format!("{text}\n")
    })
}

/// One file's text, the way `Path.read_text(encoding="utf-8")` reads it: UTF-8
/// decoded, and the newlines it ends lines with translated to `\n`.
fn read_text(path: &Path) -> PyResult<String> {
    let bytes = fs::read(path).map_err(|error| PyOSError::new_err(error.to_string()))?;
    let text = String::from_utf8(bytes)
        .map_err(|error| PyUnicodeDecodeError::new_err(error.to_string()))?;
    Ok(if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text
    })
}

/// (before, code, after): a file's code, and the comments around it lifted.
///
/// A comment above the first line of code or below the last one is lifted only
/// when it reads back as the same line; one that would not stays in the code
/// block, so the section reads back as the file, line for line. A comment
/// between two lines of code stays with them.
fn lifted(text: &str) -> (String, String, String) {
    let lines = markdown::split_lines(text);
    let code_at: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_index, line)| code_line(line))
        .map(|(index, _line)| index)
        .collect();
    if code_at.is_empty() {
        if lines.iter().all(|line| liftable(line)) {
            return (prose(&lines), String::new(), String::new());
        }
        return (String::new(), code(&lines), String::new());
    }
    let first = code_at[0];
    let last = code_at[code_at.len() - 1];
    let mut start = 0;
    while start < first && liftable(lines[start]) {
        start += 1;
    }
    let mut end = lines.len();
    while end > last + 1 && liftable(lines[end - 1]) {
        end -= 1;
    }
    (
        prose(&lines[..start]),
        code(&lines[start..end]),
        prose(&lines[end..]),
    )
}

fn code_line(line: &str) -> bool {
    !line.trim().is_empty() && !line.trim_start().starts_with('#')
}

/// Does this line read back as itself when it is written as prose?
fn liftable(line: &str) -> bool {
    if line.trim().is_empty() {
        return line.is_empty(); // a blank line is a blank line
    }
    if !line.trim_start().starts_with('#') {
        return false;
    }
    let text = prose_text(line);
    if markdown::match_heading(&text).is_some() || markdown::match_fence(&text).is_some() {
        return false; // it would name a file, or open a block
    }
    markdown::comment(&text) == line
}

/// One comment line as prose: `# ` gone, the rest as the line has it.
fn prose_text(line: &str) -> String {
    let stripped = line.trim_start();
    let mut chars = stripped.chars();
    chars.next();
    let text = chars.as_str();
    match text.strip_prefix(' ') {
        Some(rest) => rest.to_string(),
        None => text.to_string(),
    }
}

/// Lines of code as the file has them: one line, one newline.
fn code(lines: &[&str]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Comment lines as prose: the `#` gone, the empty ones kept as breaks.
fn prose(lines: &[&str]) -> String {
    lines
        .iter()
        .map(|line| prose_text(line))
        .collect::<Vec<String>>()
        .join("\n")
}

/// Backticks that no line of the code closes: three, unless the code needs more.
fn fence_for(code: &str) -> String {
    let mut longest = 0;
    for line in markdown::split_lines(code) {
        if let Some(opened) = markdown::match_fence(line) {
            if opened.marker.starts_with('`') {
                longest = longest.max(opened.marker.len());
            }
        }
    }
    "`".repeat(usize::max(3, longest + 1))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viba_repomix, m)?)?;
    Ok(())
}

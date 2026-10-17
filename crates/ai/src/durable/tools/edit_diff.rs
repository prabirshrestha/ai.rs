//! Port of durable `src/tools/edit-diff.ts`: shared diff computation utilities for the edit and similar tools.
//!
//! Divergences from Pi:
//! - Offsets and lengths are UTF-8 byte offsets instead of UTF-16 code units; every offset is computed and used in
//!   the same string, so the matched regions are the same.
//! - jsdiff's `diffLines` is `similar`'s Myers line diff, grouped into jsdiff's change objects (removed before
//!   added). Where the two Myers implementations break ties differently, hunks can differ while still describing
//!   the same edit. `createTwoFilesPatch` (with `FILE_HEADERS_ONLY`) is ported over those change objects.
//! - `trimEnd()` uses JS's whitespace set ([`is_js_whitespace`]), not Rust's `char::is_whitespace`.

use similar::{Algorithm, ChangeTag, TextDiff};
use unicode_normalization::UnicodeNormalization;

use crate::durable::errors::{Error, Result};

/// `"\r\n" | "\n"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    CrLf,
}

impl LineEnding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lf => "\n",
            Self::CrLf => "\r\n",
        }
    }
}

pub fn detect_line_ending(content: &str) -> LineEnding {
    let crlf_idx = content.find("\r\n");
    let lf_idx = content.find('\n');
    match (crlf_idx, lf_idx) {
        (_, None) | (None, _) => LineEnding::Lf,
        (Some(crlf), Some(lf)) => {
            if crlf < lf {
                LineEnding::CrLf
            } else {
                LineEnding::Lf
            }
        }
    }
}

pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::CrLf => text.replace('\n', "\r\n"),
        LineEnding::Lf => text.to_string(),
    }
}

/// JS `WhiteSpace` and `LineTerminator` (what `String.prototype.trim*` strips).
pub fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// Normalize text for fuzzy matching. Applies progressive transformations:
/// - Strip trailing whitespace from each line
/// - Normalize smart quotes to ASCII equivalents
/// - Normalize Unicode dashes/hyphens to ASCII hyphen
/// - Normalize special Unicode spaces to regular space
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    nfkc.split('\n')
        .map(|line| line.trim_end_matches(is_js_whitespace))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|c| match c {
            // Smart single quotes → '
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            // Smart double quotes → "
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            // Various dashes/hyphens → -
            // U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash,
            // U+2013 en-dash, U+2014 em-dash, U+2015 horizontal bar, U+2212 minus
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            // Special spaces → regular space
            // U+00A0 NBSP, U+2002-U+200A various spaces, U+202F narrow NBSP,
            // U+205F medium math space, U+3000 ideographic space
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            c => c,
        })
        .collect()
}

/// `content.match(/[^\n]*\n|[^\n]+/g) ?? []`.
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    content.split_inclusive('\n').collect()
}

#[derive(Debug, Clone, Copy)]
struct LineSpan {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

/// `Pick<MatchedEdit, "matchIndex" | "matchLength" | "newText">`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextReplacement {
    pub match_index: usize,
    pub match_length: usize,
    pub new_text: String,
}

fn get_line_spans(content: &str) -> Vec<LineSpan> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = LineSpan {
                start: offset,
                end: offset + line.len(),
            };
            offset = span.end;
            span
        })
        .collect()
}

fn get_replacement_line_range(
    lines: &[LineSpan],
    replacement: &TextReplacement,
) -> Result<(usize, usize)> {
    let replacement_start = replacement.match_index;
    let replacement_end = replacement.match_index + replacement.match_length;

    let start_line = lines
        .iter()
        .position(|line| replacement_start >= line.start && replacement_start < line.end)
        .ok_or_else(|| Error::message("Replacement range is outside the base content."))?;

    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < replacement_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        return Err(Error::message(
            "Replacement range is outside the base content.",
        ));
    }

    Ok((start_line, end_line + 1))
}

fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
    let mut result = content.to_string();
    for replacement in replacements.iter().rev() {
        let match_index = replacement.match_index - offset;
        result = format!(
            "{}{}{}",
            &result[..match_index],
            replacement.new_text,
            &result[match_index + replacement.match_length..]
        );
    }
    result
}

/// Apply replacements matched against `base_content` to `original_content` while
/// preserving unchanged line blocks from the original.
///
/// This is useful when `base_content` is a normalized view of the original. Each
/// replacement is widened to the lines it actually touches, those touched lines
/// are rewritten from the normalized base, and all other lines are copied back
/// from `original_content`. The actual replacement ranges drive preservation so
/// duplicate normalized lines cannot be aligned to the wrong occurrence.
pub fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[TextReplacement],
) -> Result<String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = get_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(Error::message(
            "Cannot preserve unchanged lines because the base content has a different line count.",
        ));
    }

    struct Group {
        start_line: usize,
        end_line: usize,
        replacements: Vec<TextReplacement>,
    }
    let mut groups: Vec<Group> = Vec::new();
    let mut sorted_replacements = replacements.to_vec();
    sorted_replacements.sort_by_key(|replacement| replacement.match_index);
    for replacement in sorted_replacements {
        let (start_line, end_line) = get_replacement_line_range(&base_lines, &replacement)?;
        if let Some(current) = groups.last_mut()
            && start_line < current.end_line
        {
            current.end_line = current.end_line.max(end_line);
            current.replacements.push(replacement);
            continue;
        }
        groups.push(Group {
            start_line,
            end_line,
            replacements: vec![replacement],
        });
    }

    let mut original_line_index = 0;
    let mut result = String::new();
    for group in &groups {
        result.push_str(&original_lines[original_line_index..group.start_line].concat());

        let group_start_offset = base_lines[group.start_line].start;
        let group_end_offset = base_lines[group.end_line - 1].end;
        result.push_str(&apply_replacements(
            &base_content[group_start_offset..group_end_offset],
            &group.replacements,
            group_start_offset,
        ));
        original_line_index = group.end_line;
    }
    result.push_str(&original_lines[original_line_index..].concat());

    Ok(result)
}

/// `FuzzyMatchResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyMatchResult {
    /// Whether a match was found
    pub found: bool,
    /// The index where the match starts (in the content that should be used for replacement); `None` is TS `-1`.
    pub index: Option<usize>,
    /// Length of the matched text
    pub match_length: usize,
    /// Whether fuzzy matching was used (false = exact match)
    pub used_fuzzy_match: bool,
    /// The content to use for replacement operations.
    /// When exact match: original content. When fuzzy match: normalized content.
    pub content_for_replacement: String,
}

/// `Edit`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// `AppliedEditsResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedEditsResult {
    pub base_content: String,
    pub new_content: String,
}

/// Find `old_text` in content, trying exact match first, then fuzzy match.
/// When fuzzy matching is used, the returned `content_for_replacement` is the
/// fuzzy-normalized version of the content (trailing whitespace stripped,
/// Unicode quotes/dashes normalized to ASCII).
pub fn fuzzy_find_text(content: &str, old_text: &str) -> FuzzyMatchResult {
    // Try exact match first
    if let Some(exact_index) = content.find(old_text) {
        return FuzzyMatchResult {
            found: true,
            index: Some(exact_index),
            match_length: old_text.len(),
            used_fuzzy_match: false,
            content_for_replacement: content.to_string(),
        };
    }

    // Try fuzzy match - work entirely in normalized space
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    let Some(fuzzy_index) = fuzzy_content.find(&fuzzy_old_text) else {
        return FuzzyMatchResult {
            found: false,
            index: None,
            match_length: 0,
            used_fuzzy_match: false,
            content_for_replacement: content.to_string(),
        };
    };

    // When fuzzy matching, return offsets in normalized space. Callers can use
    // the normalized content to compute replacements, then decide how much of
    // that normalized output should be written back.
    FuzzyMatchResult {
        found: true,
        index: Some(fuzzy_index),
        match_length: fuzzy_old_text.len(),
        used_fuzzy_match: true,
        content_for_replacement: fuzzy_content,
    }
}

/// Strip UTF-8 BOM if present, return both the BOM (if any) and the text without it.
pub fn strip_bom(content: &str) -> (&'static str, &str) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}", text),
        None => ("", content),
    }
}

/// `fuzzyContent.split(fuzzyOldText).length - 1`.
fn count_occurrences(content: &str, old_text: &str) -> i64 {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    if fuzzy_old_text.is_empty() {
        // `split("")` splits into UTF-16 code units, and `"".split("")` is `[]`.
        return fuzzy_content.encode_utf16().count() as i64 - 1;
    }
    fuzzy_content.matches(fuzzy_old_text.as_str()).count() as i64
}

fn get_not_found_error(path: &str, edit_index: usize, total_edits: usize) -> Error {
    if total_edits == 1 {
        return Error::message(format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        ));
    }
    Error::message(format!(
        "Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
    ))
}

fn get_duplicate_error(
    path: &str,
    edit_index: usize,
    total_edits: usize,
    occurrences: i64,
) -> Error {
    if total_edits == 1 {
        return Error::message(format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        ));
    }
    Error::message(format!(
        "Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
    ))
}

fn get_empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> Error {
    if total_edits == 1 {
        return Error::message(format!("oldText must not be empty in {path}."));
    }
    Error::message(format!(
        "edits[{edit_index}].oldText must not be empty in {path}."
    ))
}

fn get_no_change_error(path: &str, total_edits: usize) -> Error {
    if total_edits == 1 {
        return Error::message(format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        ));
    }
    Error::message(format!(
        "No changes made to {path}. The replacements produced identical content."
    ))
}

/// Apply one or more exact-text replacements to LF-normalized content.
///
/// All edits are matched against the same original content. Replacements are
/// then applied in reverse order so offsets remain stable. If any edit needs
/// fuzzy matching, the operation runs in fuzzy-normalized content space and then
/// overlays those line-level changes onto the original content so unchanged line
/// blocks keep their original bytes.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEditsResult> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();

    for (i, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(get_empty_old_text_error(path, i, normalized_edits.len()));
        }
    }

    let used_fuzzy_match = normalized_edits
        .iter()
        .any(|edit| fuzzy_find_text(normalized_content, &edit.old_text).used_fuzzy_match);
    let replacement_base_content = if used_fuzzy_match {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        normalized_content.to_string()
    };

    let mut matched_edits: Vec<MatchedEdit> = Vec::new();
    for (i, edit) in normalized_edits.iter().enumerate() {
        let match_result = fuzzy_find_text(&replacement_base_content, &edit.old_text);
        let Some(match_index) = match_result.index.filter(|_| match_result.found) else {
            return Err(get_not_found_error(path, i, normalized_edits.len()));
        };

        let occurrences = count_occurrences(&replacement_base_content, &edit.old_text);
        if occurrences > 1 {
            return Err(get_duplicate_error(
                path,
                i,
                normalized_edits.len(),
                occurrences,
            ));
        }

        matched_edits.push(MatchedEdit {
            edit_index: i,
            match_index,
            match_length: match_result.match_length,
            new_text: edit.new_text.clone(),
        });
    }

    matched_edits.sort_by_key(|edit| edit.match_index);
    for pair in matched_edits.windows(2) {
        let (previous, current) = (&pair[0], &pair[1]);
        if previous.match_index + previous.match_length > current.match_index {
            return Err(Error::message(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                previous.edit_index, current.edit_index
            )));
        }
    }

    let replacements: Vec<TextReplacement> = matched_edits
        .into_iter()
        .map(|edit| TextReplacement {
            match_index: edit.match_index,
            match_length: edit.match_length,
            new_text: edit.new_text,
        })
        .collect();
    let base_content = normalized_content.to_string();
    let new_content = if used_fuzzy_match {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base_content,
            &replacements,
        )?
    } else {
        apply_replacements(&replacement_base_content, &replacements, 0)
    };

    if base_content == new_content {
        return Err(get_no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEditsResult {
        base_content,
        new_content,
    })
}

/// One jsdiff change object of `diffLines`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Part {
    value: String,
    added: bool,
    removed: bool,
}

/// jsdiff `diffLines(old, new)`: changed runs list their removed lines before their added lines.
fn diff_lines(old_content: &str, new_content: &str) -> Vec<Part> {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_lines(old_content, new_content);
    let mut parts: Vec<Part> = Vec::new();
    let mut equal = String::new();
    let mut removed = String::new();
    let mut added = String::new();
    let flush_changes = |parts: &mut Vec<Part>, removed: &mut String, added: &mut String| {
        if !removed.is_empty() {
            parts.push(Part {
                value: std::mem::take(removed),
                added: false,
                removed: true,
            });
        }
        if !added.is_empty() {
            parts.push(Part {
                value: std::mem::take(added),
                added: true,
                removed: false,
            });
        }
    };
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Equal => {
                flush_changes(&mut parts, &mut removed, &mut added);
                equal.push_str(change.value());
            }
            tag => {
                if !equal.is_empty() {
                    parts.push(Part {
                        value: std::mem::take(&mut equal),
                        added: false,
                        removed: false,
                    });
                }
                if tag == ChangeTag::Delete {
                    removed.push_str(change.value());
                } else {
                    added.push_str(change.value());
                }
            }
        }
    }
    flush_changes(&mut parts, &mut removed, &mut added);
    if !equal.is_empty() {
        parts.push(Part {
            value: equal,
            added: false,
            removed: false,
        });
    }
    parts
}

/// jsdiff's `splitLines`: lines with their trailing newline, where present.
fn split_patch_lines(text: &str) -> Vec<String> {
    let has_trailing_nl = text.ends_with('\n');
    let mut result: Vec<String> = text.split('\n').map(|line| format!("{line}\n")).collect();
    if has_trailing_nl {
        result.pop();
    } else if let Some(last) = result.last_mut() {
        last.pop();
    }
    result
}

struct Hunk {
    old_start: usize,
    old_lines: usize,
    new_start: usize,
    new_lines: usize,
    lines: Vec<String>,
}

/// jsdiff `structuredPatch(...).hunks` over `diffLines` changes.
fn structured_hunks(old_content: &str, new_content: &str, context: usize) -> Vec<Hunk> {
    let mut diff: Vec<(Part, Vec<String>)> = diff_lines(old_content, new_content)
        .into_iter()
        .map(|part| {
            let lines = split_patch_lines(&part.value);
            (part, lines)
        })
        .collect();
    // Append an empty value to make cleanup easier
    diff.push((
        Part {
            value: String::new(),
            added: false,
            removed: false,
        },
        Vec::new(),
    ));
    let context_lines =
        |lines: &[String]| -> Vec<String> { lines.iter().map(|line| format!(" {line}")).collect() };

    let mut hunks: Vec<Hunk> = Vec::new();
    let (mut old_range_start, mut new_range_start) = (0usize, 0usize);
    let mut cur_range: Vec<String> = Vec::new();
    let (mut old_line, mut new_line) = (1usize, 1usize);
    for i in 0..diff.len() {
        let (current, lines) = &diff[i];
        if current.added || current.removed {
            if old_range_start == 0 {
                old_range_start = old_line;
                new_range_start = new_line;
                if i > 0 {
                    let prev = &diff[i - 1].1;
                    cur_range = if context > 0 {
                        context_lines(&prev[prev.len().saturating_sub(context)..])
                    } else {
                        Vec::new()
                    };
                    old_range_start -= cur_range.len();
                    new_range_start -= cur_range.len();
                }
            }
            for line in lines {
                cur_range.push(format!("{}{line}", if current.added { '+' } else { '-' }));
            }
            if current.added {
                new_line += lines.len();
            } else {
                old_line += lines.len();
            }
        } else {
            if old_range_start != 0 {
                if lines.len() <= context * 2 && i + 2 < diff.len() {
                    cur_range.extend(context_lines(lines));
                } else {
                    let context_size = lines.len().min(context);
                    cur_range.extend(context_lines(&lines[..context_size]));
                    hunks.push(Hunk {
                        old_start: old_range_start,
                        old_lines: old_line - old_range_start + context_size,
                        new_start: new_range_start,
                        new_lines: new_line - new_range_start + context_size,
                        lines: std::mem::take(&mut cur_range),
                    });
                    old_range_start = 0;
                    new_range_start = 0;
                }
            }
            old_line += lines.len();
            new_line += lines.len();
        }
    }
    for hunk in &mut hunks {
        let mut lines = Vec::with_capacity(hunk.lines.len());
        for line in hunk.lines.drain(..) {
            match line.strip_suffix('\n') {
                Some(stripped) => lines.push(stripped.to_string()),
                None => {
                    lines.push(line);
                    lines.push("\\ No newline at end of file".to_string());
                }
            }
        }
        hunk.lines = lines;
    }
    hunks
}

/// Generate a standard unified patch (`createTwoFilesPatch` with `FILE_HEADERS_ONLY`).
pub fn generate_unified_patch(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: Option<usize>,
) -> String {
    let context = context_lines.unwrap_or(4);
    let mut ret = vec![format!("--- {path}"), format!("+++ {path}")];
    for hunk in structured_hunks(old_content, new_content, context) {
        let old_start = if hunk.old_lines == 0 {
            hunk.old_start - 1
        } else {
            hunk.old_start
        };
        let new_start = if hunk.new_lines == 0 {
            hunk.new_start - 1
        } else {
            hunk.new_start
        };
        ret.push(format!(
            "@@ -{old_start},{} +{new_start},{} @@",
            hunk.old_lines, hunk.new_lines
        ));
        ret.extend(hunk.lines);
    }
    ret.join("\n") + "\n"
}

/// What [`generate_diff_string`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffString {
    pub diff: String,
    /// The first changed line number (in the new file).
    pub first_changed_line: Option<usize>,
}

/// Context lines shown by [`generate_diff_string`], numbered in the old file; both counters advance.
fn push_context(
    output: &mut Vec<String>,
    lines: &[&str],
    width: usize,
    old_line_num: &mut usize,
    new_line_num: &mut usize,
) {
    for line in lines {
        output.push(format!(" {:>width$} {line}", *old_line_num));
        *old_line_num += 1;
        *new_line_num += 1;
    }
}

/// Generate a display-oriented diff string with line numbers and context.
/// Returns both the diff string and the first changed line number (in the new file).
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: Option<usize>,
) -> DiffString {
    let context_lines = context_lines.unwrap_or(4);
    let parts = diff_lines(old_content, new_content);
    let mut output: Vec<String> = Vec::new();

    let old_lines = old_content.split('\n').count();
    let new_lines = new_content.split('\n').count();
    let max_line_num = old_lines.max(new_lines);
    let width = max_line_num.to_string().len();
    let ellipsis = format!(" {} ...", " ".repeat(width));

    let mut old_line_num = 1;
    let mut new_line_num = 1;
    let mut last_was_change = false;
    let mut first_changed_line: Option<usize> = None;

    for (i, part) in parts.iter().enumerate() {
        let mut raw: Vec<&str> = part.value.split('\n').collect();
        if raw.last() == Some(&"") {
            raw.pop();
        }

        if part.added || part.removed {
            // Capture the first changed line (in the new file)
            if first_changed_line.is_none() {
                first_changed_line = Some(new_line_num);
            }

            // Show the change
            for line in raw {
                if part.added {
                    output.push(format!("+{new_line_num:>width$} {line}"));
                    new_line_num += 1;
                } else {
                    // removed
                    output.push(format!("-{old_line_num:>width$} {line}"));
                    old_line_num += 1;
                }
            }
            last_was_change = true;
        } else {
            // Context lines - only show a few before/after changes
            let next_part_is_change = parts
                .get(i + 1)
                .is_some_and(|next| next.added || next.removed);
            let has_leading_change = last_was_change;
            let has_trailing_change = next_part_is_change;

            if has_leading_change && has_trailing_change {
                if raw.len() <= context_lines * 2 {
                    push_context(
                        &mut output,
                        &raw,
                        width,
                        &mut old_line_num,
                        &mut new_line_num,
                    );
                } else {
                    let leading_lines = &raw[..context_lines];
                    let trailing_lines = &raw[raw.len() - context_lines..];
                    let skipped_lines = raw.len() - leading_lines.len() - trailing_lines.len();

                    push_context(
                        &mut output,
                        leading_lines,
                        width,
                        &mut old_line_num,
                        &mut new_line_num,
                    );

                    output.push(ellipsis.clone());
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;

                    push_context(
                        &mut output,
                        trailing_lines,
                        width,
                        &mut old_line_num,
                        &mut new_line_num,
                    );
                }
            } else if has_leading_change {
                let shown_lines = &raw[..raw.len().min(context_lines)];
                let skipped_lines = raw.len() - shown_lines.len();

                push_context(
                    &mut output,
                    shown_lines,
                    width,
                    &mut old_line_num,
                    &mut new_line_num,
                );

                if skipped_lines > 0 {
                    output.push(ellipsis.clone());
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;
                }
            } else if has_trailing_change {
                let skipped_lines = raw.len().saturating_sub(context_lines);
                if skipped_lines > 0 {
                    output.push(ellipsis.clone());
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;
                }

                push_context(
                    &mut output,
                    &raw[skipped_lines..],
                    width,
                    &mut old_line_num,
                    &mut new_line_num,
                );
            } else {
                // Skip these context lines entirely
                old_line_num += raw.len();
                new_line_num += raw.len();
            }

            last_was_change = false;
        }
    }

    DiffString {
        diff: output.join("\n"),
        first_changed_line,
    }
}

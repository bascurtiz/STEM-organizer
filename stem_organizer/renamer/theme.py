"""Renamer theme tokens + tips.

Tokens are shared with the rest of the app via ``stem_organizer.theme.DARK``;
we re-export them here so the renamer module reads like the original
track_renamer.gui.theme.
"""
from __future__ import annotations

from .. import theme

DARK = dict(theme.DARK)
PREVIEW_LOG_FONT_FAMILY = theme.FONT_FAMILY_MONO
PREVIEW_LOG_FONT_SIZE = 9
PREVIEW_LOG_PCT_FONT_SIZE = 8


TIPS = {
    "preset": "Load a saved rule preset.",
    "save_preset": "Save the current rule stack as a preset.",
    "open_folder": "Scan a folder for audio and MIDI files to rename.",
    "help": "How rules work and supported file types.",
    "delete_preset": "Delete the selected template.",
    "recursive": "Also scan files inside subfolders of the selected directory.",
    "apply_preview": "Update the preview list with the current rules. Does not run Auto-detect — use Analyze for that.",
    "clear_rules": "Remove all rules and reset the list.",
    "add_rule": "Choose a renaming rule to add to the stack.",
    "condition_field": "Which part of the filename to test.",
    "condition_op": "How to compare the filename against your value.",
    "condition_value": "Text or pattern to match. Supports tokens like {group} or {bpm}.",
    "rule_enable": "Enable or disable this specific modification.",
    "remove_rule": "Remove this rule from the stack.",
    "remove_category_row": "Remove this category mapping row.",
    "add_category_row": "Add a new category mapping row.",
    "prefix_field": "Text prepended when keywords in this row match.",
    "keywords_field": "Comma-separated keywords that trigger this prefix (plurals match automatically).",
    "category_color": "Click to change this category's color.",
    "instrument_source": "How the category PREFIX is chosen for each file.",
    "instrument_source_filename": "Filename: match keywords in the current name against the category rows below. No audio listening — preview updates immediately with Apply.",
    "instrument_source_auto": "Audio: classify the audio with ML (PaSST OpenMIC-2018, 20 instruments) and map the top label to a category PREFIX.",
    "instrument_source_combo": "Combo: try Filename keywords first; if none match, fall back to Audio on that file's audio.",
    "add_child_rule": "Add another operation inside this condition group.",
    "select_all": "Select every file in the preview for renaming.",
    "deselect_all": "Deselect every file in the preview.",
    "only_changed": "Show only files that will be modified by the current rules.",
    "file_checkbox": "Include or exclude this file when applying renames.",
    "detected_keyword": "Filename keyword that matched, or <audio-determined> when the category came from ML listening.",
    "change_prefix": (
        "Shift-click or Ctrl-click to multi-select rows,\n"
        "then right-click → Change to: to replace the category prefix and rename on disk."
    ),
    "cancel": "Stop Analyze (Auto-detect) or clear the active preview selection.",
    "rename": "Permanently rename the selected files on disk.",
    "analyze": (
        "Run Auto-detect (PaSST OpenMIC) on the selected files, then confirm rename. "
        "Preview Apply does not analyze audio — only this button does."
    ),
    "remove_text": "Text to remove from the filename.",
    "rule_text": "Text used by this rule (remove / replace / add).",
    "play_preview": "Play / pause the selected preview file.",
    "close_dialog": "Close this dialog.",
    "ok_dialog": "Confirm and continue.",
    "cancel_dialog": "Dismiss without saving.",
    # Radio values are filename / model / combo (UI label for model = Audio).
    "instrument_source_model": (
        "Audio: classify the audio with ML (PaSST OpenMIC-2018, 20 instruments) "
        "and map the top label to a category PREFIX."
    ),
}

TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}

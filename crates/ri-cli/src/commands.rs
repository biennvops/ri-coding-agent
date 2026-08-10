use ri_core::AppState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommandKind {
    Model,
    New,
    Resume,
    Name,
    Session,
    Compact,
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommandArgument {
    None,
    Optional(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CommandSpec {
    pub kind: CommandKind,
    pub name: &'static str,
    pub description: &'static str,
    pub argument: CommandArgument,
}

pub(crate) const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        kind: CommandKind::Model,
        name: "model",
        description: "Select model",
        argument: CommandArgument::Optional("provider/model"),
    },
    CommandSpec {
        kind: CommandKind::New,
        name: "new",
        description: "Start new session",
        argument: CommandArgument::None,
    },
    CommandSpec {
        kind: CommandKind::Resume,
        name: "resume",
        description: "Resume session",
        argument: CommandArgument::None,
    },
    CommandSpec {
        kind: CommandKind::Name,
        name: "name",
        description: "Show or set session name",
        argument: CommandArgument::Optional("name"),
    },
    CommandSpec {
        kind: CommandKind::Session,
        name: "session",
        description: "Show session details",
        argument: CommandArgument::None,
    },
    CommandSpec {
        kind: CommandKind::Compact,
        name: "compact",
        description: "Compact current context",
        argument: CommandArgument::None,
    },
    CommandSpec {
        kind: CommandKind::Quit,
        name: "quit",
        description: "Exit ri",
        argument: CommandArgument::None,
    },
];

pub(crate) fn command_spec(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|spec| spec.name == name)
}

pub(crate) fn matching_commands(input: &str) -> impl Iterator<Item = &'static CommandSpec> + '_ {
    let prefix = command_prefix(input);
    COMMANDS.iter().filter(move |spec| {
        prefix
            .as_ref()
            .is_some_and(|prefix| spec.name.starts_with(prefix))
    })
}

fn command_prefix(input: &str) -> Option<&str> {
    let input = input.strip_prefix('/')?;
    (!input.chars().any(char::is_whitespace)).then_some(input)
}

pub(crate) fn command_help() -> String {
    COMMANDS
        .iter()
        .map(|spec| {
            let usage = match spec.argument {
                CommandArgument::None => format!("/{}", spec.name),
                CommandArgument::Optional(argument) => format!("/{} [{argument}]", spec.name),
            };
            format!("  {usage:<32} {}", spec.description)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CommandSuggestions {
    selected: usize,
    dismissed_revision: Option<u64>,
}

impl CommandSuggestions {
    pub fn is_visible(&self, state: &AppState) -> bool {
        !state.is_busy()
            && self.dismissed_revision != Some(state.input_revision())
            && matching_commands(state.input()).next().is_some()
    }

    pub fn selected(&self, state: &AppState) -> usize {
        let count = matching_commands(state.input()).count();
        self.selected.min(count.saturating_sub(1))
    }

    pub fn move_up(&mut self, state: &AppState) {
        let count = matching_commands(state.input()).count();
        if count > 0 {
            let selected = self.selected(state);
            self.selected = selected.checked_sub(1).unwrap_or(count - 1);
        }
    }

    pub fn move_down(&mut self, state: &AppState) {
        let count = matching_commands(state.input()).count();
        if count > 0 {
            self.selected = (self.selected(state) + 1) % count;
        }
    }

    pub fn complete(&mut self, state: &mut AppState) -> bool {
        let Some(spec) = self.selected_spec(state) else {
            return false;
        };
        let suffix = match spec.argument {
            CommandArgument::None => "",
            CommandArgument::Optional(_) => " ",
        };
        state.set_input(format!("/{}{suffix}", spec.name));
        true
    }

    pub fn accept(&self, state: &mut AppState) -> bool {
        let Some(spec) = self.selected_spec(state) else {
            return false;
        };
        state.set_input(format!("/{}", spec.name));
        true
    }

    fn selected_spec(&self, state: &AppState) -> Option<&'static CommandSpec> {
        self.is_visible(state)
            .then(|| matching_commands(state.input()).nth(self.selected(state)))
            .flatten()
    }

    pub fn dismiss(&mut self, state: &AppState) {
        if self.is_visible(state) {
            self.dismissed_revision = Some(state.input_revision());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(input: &str) -> Vec<&'static str> {
        matching_commands(input).map(|spec| spec.name).collect()
    }

    #[test]
    fn registry_contains_every_supported_command() {
        assert_eq!(
            COMMANDS.iter().map(|spec| spec.name).collect::<Vec<_>>(),
            ["model", "new", "resume", "name", "session", "compact", "quit"]
        );
    }

    #[test]
    fn suggestions_match_command_prefixes_without_fuzzy_search() {
        assert_eq!(names("/").len(), COMMANDS.len());
        assert_eq!(names("/m"), ["model"]);
        assert!(names("/z").is_empty());
        assert!(names(" /m").is_empty());
        assert!(names("/model argument").is_empty());
    }

    #[test]
    fn suggestion_selection_wraps_and_completion_uses_argument_metadata() {
        let mut state = AppState::new();
        state.insert_text("/");
        let mut suggestions = CommandSuggestions::default();

        suggestions.move_up(&state);
        assert_eq!(suggestions.selected(&state), COMMANDS.len() - 1);
        suggestions.move_down(&state);
        assert_eq!(suggestions.selected(&state), 0);
        assert!(suggestions.complete(&mut state));
        assert_eq!(state.input(), "/model ");

        state.set_input("/".to_owned());
        suggestions.move_up(&state);
        assert!(suggestions.complete(&mut state));
        assert_eq!(state.input(), "/quit");
    }

    #[test]
    fn enter_accepts_the_highlighted_command_without_argument_spacing() {
        let mut state = AppState::new();
        state.insert_text("/");
        let mut suggestions = CommandSuggestions::default();
        suggestions.move_down(&state);

        assert!(suggestions.accept(&mut state));
        assert_eq!(state.input(), "/new");
    }

    #[test]
    fn escape_dismisses_until_the_input_changes() {
        let mut state = AppState::new();
        state.insert_text("/");
        let mut suggestions = CommandSuggestions::default();
        assert!(suggestions.is_visible(&state));

        suggestions.dismiss(&state);
        assert!(!suggestions.is_visible(&state));
        assert_eq!(state.input(), "/");
        state.insert_text("m");
        assert!(suggestions.is_visible(&state));
    }
}
